__version__ = "1.0"

from re import M
from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL

class DepthAnythingV3NodeSize(desc.MultiDynamicNodeSize):
    def computeSize(self, node):
        if node.attribute(self._params[0]).isLink:
            return node.attribute(self._params[0]).inputLink.node.size

        from pathlib import Path

        input_path_param = node.attribute(self._params[0])
        extension_param = node.attribute(self._params[1])
        input_path = input_path_param.value
        extension = extension_param.value
        include_suffixes = [extension.lower(), extension.upper()]

        size = 1
        if Path(input_path).is_dir():
            import itertools
            image_paths = list(itertools.chain(*(Path(input_path).glob(f'*.{suffix}') for suffix in include_suffixes)))
            size = len(image_paths)
        
        return size

class DepthAnythingV3BlockSize(desc.Parallelization):
    def getSizes(self, node):
        import math

        size = node.size
        if node.attribute('blockSize').value:
            nbBlocks = int(math.ceil(float(size) / float(node.attribute('blockSize').value)))
            return node.attribute('blockSize').value, size, nbBlocks
        else:
            return size, size, 1


class DepthAnythingV3(desc.Node):
    category = "Image Intrinsics"
    documentation = """This node computes depth, from a monocular image using the DepthAnythingV3 deep model."""
    
    gpu = desc.Level.INTENSIVE

    size = DepthAnythingV3NodeSize(['inputImages', 'inputExtension'])
    parallelization = DepthAnythingV3BlockSize()

    inputs = [
        desc.File(
            name="inputImages",
            label="Input Images",
            description="Input images to process. Folder path or sfmData filepath",
            value="",
        ),
        desc.ChoiceParam(
            name="inputExtension",
            label="Input Extension",
            description="Extension of the input images. This will be used to determine which images are to be used if \n"
                        "a directory is provided as the input.",
            values=["jpg", "jpeg", "png", "exr"],
            value="exr",
            exclusive=True,
        ),
        desc.ChoiceParam(
            name="sam3Model",
            label="SAM3 Model",
            description="SAM3 model to be used.\n"
                        "Be aware that the Nested-Giant-Large model has a NON-Commercial license CC BY-NC 4.0.",
            values=["Base", "Metric-Large", "Nested-Giant-Large"],
            value="Metric-Large",
            exclusive=True,
        ),
        desc.FloatParam(
            name="focalpix",
            label="Focal in pixels",
            value=300.0,
            description="Focal value in pixels used if it cannot be extracted from metadata.",
            range=(1.0, 10000.0, 1.0),
            enabled=lambda node: node.sam3Model.value == "Metric-Large",
        ),
        desc.BoolParam(
            name="useCameraPoseIfAvailable",
            label="Use Camera Pose If Available",
            description="Feed the model with intrinsics and extrinsics extracted from sfmData.",
            value=True,
            enabled=lambda node: node.inputImages.value[-4:].lower() in [".sfm", ".abc"] and node.sam3Model.value != "Metric-Large",
        ),
        desc.BoolParam(
            name="outputDepth",
            label="Output Depth Map",
            description="If this option is enabled, a depth map is generated.",
            value=True,
        ),
        desc.BoolParam(
            name="saveVisuImages",
            label="Save images for visualization",
            description="Save additional png images for depth map.",
            value=False,
        ),
        desc.IntParam(
            name="blockSize",
            label="Block Size",
            value=50,
            description="Sets the number of images to process in one chunk. If set to 0, all images are processed at once.",
            range=(0, 1000, 1),
        ),
        desc.ChoiceParam(
            name="verboseLevel",
            label="Verbose Level",
            description="Verbosity level (fatal, error, warning, info, debug, trace).",
            values=VERBOSE_LEVEL,
            value="info",
        ),
        desc.BoolParam(
            name="inputNameIsViewId",
            label="ViewIds As Input Name",
            description="Enabled output visu when input images are named with viewIds.",
            value=False,
            advanced=True,
            invalidate=False,
            enabled=lambda node: node.inputImages.value[-4:].lower() in [".sfm", ".abc"],
        ),
    ]

    outputs = [
        desc.File(
            name='output',
            label='Output Folder',
            description="Output folder containing the depth maps saved as exr images.",
            value="{nodeCacheFolder}",
        ),
        desc.File(
            name="DepthMap",
            label="Depth Map",
            description="Output depth map",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/depth_<VIEW_ID>.exr" if attr.node.inputNameIsViewId.value else "{nodeCacheFolder}/depth_<FILESTEM>.exr",
            group="",
            enabled=lambda node: node.outputDepth.value,
        ),
        desc.File(
            name="DepthMapColor",
            label="Colored Depth Map",
            description="Output colored depth map",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/depth_vis_<VIEW_ID>.png" if attr.node.inputNameIsViewId.value else "{nodeCacheFolder}/depth_vis_<FILESTEM>.png",
            group="",
            enabled=lambda node: node.outputDepth.value and node.saveVisuImages.value,
        ),
    ]

    def preprocess(self, node):
        extension = node.inputExtension.value
        input_path = node.inputImages.value

        image_paths = get_image_paths_list(input_path, extension)

        if len(image_paths) == 0:
            raise FileNotFoundError(f'No image files found in {input_path}')

        self.image_paths = image_paths

    def processChunk(self, chunk):

        from depth_anything_3.api import DepthAnything3
        from pyalicevision import image as avimg

        import torch
        from img_proc import image
        import json
        import os
        import numpy as np
        from pathlib import Path
        import OpenImageIO as oiio

        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            if not chunk.node.inputImages.value:
                chunk.logger.warning('No input folder given.')

            chunk_image_paths = self.image_paths[chunk.range.start:chunk.range.end]

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # computation
            chunk.logger.info(f'Starting computation on chunk {chunk.range.iteration + 1}/{chunk.range.fullSize // chunk.range.blockSize + int(chunk.range.fullSize != chunk.range.blockSize)}...')

            # Initialize models
            chunk.logger.info("Loading DepthAnything model...")

            model_name = "models--depth-anything--DA3" + chunk.node.sam3Model.value.upper()

            pretrained_path = os.getenv('DEPTH_ANYTHING_3_MODELS_PATH') + '/' + model_name
            model = DepthAnything3.from_pretrained(pretrained_path)
            model = model.to(device=device)
                
            metadata_deep_model = {}
            metadata_deep_model["Meshroom:mrImageIntrinsicsDecomposition:DeepModelName"] = "DepthAnything3-" + chunk.node.sam3Model.value
            metadata_deep_model["Meshroom:mrImageIntrinsicsDecomposition:DeepModelVersion"] = "1.1"

            images = []
            intrinsics = []
            extrinsics = []

            for idx, path in enumerate(chunk_image_paths):
                img, h_ori, w_ori, pixelAspectRatio, orientation = image.loadImage(str(chunk_image_paths[idx][0]), applyPAR = True)

                img_pil = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
                images.append(img_pil)
                if chunk_image_paths[idx][3].size and chunk.node.sam3Model.value != "Metric-Large" and chunk.node.useCameraPoseIfAvailable.value:
                    intrinsics.append(chunk_image_paths[idx][2])
                    extrinsics.append(chunk_image_paths[idx][3])

            with torch.no_grad():
                if len(intrinsics) > 0:
                    chunk.logger.info("Compute depth maps using camera poses")
                    prediction = model.inference(image=images, extrinsics=extrinsics, intrinsics=intrinsics)
                else:
                    prediction = model.inference(images,)

            # Write outputs
            for idx, path in enumerate(chunk_image_paths):

                depth = prediction.depth[idx]

                if chunk.node.sam3Model.value == "Metric-Large":
                    fpix = chunk_image_paths[idx][1] if chunk_image_paths[idx][1] is not None else 0.0
                    depth = (fpix if fpix > 0.0 else chunk.node.focalpix.value) * depth / 300

                outputDirPath = Path(chunk.node.output.value)
                image_stem = Path(chunk_image_paths[idx][0]).stem

                image_stem = str(image_stem)

                vis_file_name = "depth_vis_" + image_stem + ".png"
                vis_file_path = str(outputDirPath / vis_file_name)
                depth_file_name = "depth_" + image_stem + ".exr"
                depth_file_path = str(outputDirPath / depth_file_name)

                optWrite = avimg.ImageWriteOptions()
                optWrite.toColorSpace(avimg.EImageColorSpace_NO_CONVERSION)

                if chunk.node.outputDepth.value:
                    depth_to_write = depth[:,:,np.newaxis]
                    optWrite.exrCompressionMethod(avimg.EImageExrCompression_stringToEnum("DWAA"))
                    optWrite.exrCompressionLevel(45)
                    image.writeImage(depth_file_path, depth_to_write, h_ori, w_ori, orientation, pixelAspectRatio, metadata_deep_model, optWrite)

                    file = oiio.ImageInput.open(depth_file_path)
                    if file:
                        spec = file.spec()
                        pixels = file.read_image()
                        file.close()

                        spec.attribute("AliceVision:downscale", 1)
                        spec.attribute("AliceVision:CArr", oiio.TypeVector, tuple(chunk_image_paths[idx][5]))
                        spec.attribute("AliceVision:iCamArr", oiio.TypeMatrix33, tuple(chunk_image_paths[idx][6]))

                        updated = oiio.ImageOutput.create(depth_file_path)
                        if updated:
                            updated.open(depth_file_path, spec)
                            updated.write_image(pixels)
                            updated.close()

                if chunk.node.outputDepth.value and chunk.node.saveVisuImages.value:
                    import matplotlib
                    depth = (depth - depth.min()) / (depth.max() - depth.min())
                    cmap = matplotlib.colormaps.get_cmap('Spectral')
                    colored_depth = cmap(depth)[:, :, :3]
                    image.writeImage(vis_file_path, colored_depth, h_ori, w_ori, orientation, pixelAspectRatio, metadata_deep_model)

            chunk.logger.info('DepthAnything3 end')
        finally:
            chunk.logManager.end()

def get_image_paths_list(input_path, extension):
    from pyalicevision import sfmData
    from pyalicevision import sfmDataIO
    from pyalicevision import camera
    from pathlib import Path
    import itertools
    import numpy as np

    include_suffixes = [extension.lower(), extension.upper()]
    image_paths = []

    if Path(input_path).is_dir():
        image_paths = sorted(itertools.chain(*(Path(input_path).glob(f'*.{suffix}') for suffix in include_suffixes)))
        image_paths = [(p, -1.0, None, None, -1.0, None, None) for p in image_paths]
    elif Path(input_path).suffix.lower() in [".sfm", ".abc"]:
        if Path(input_path).exists():
            dataAV = sfmData.SfMData()
            if sfmDataIO.load(dataAV, input_path, sfmDataIO.ALL):
                views = dataAV.getViews()
                for id, v in views.items():
                    intrinsicId = v.getIntrinsicId()
                    intrinsic = dataAV.getIntrinsic(intrinsicId)
                    scaleOffset = camera.IntrinsicScaleOffset.cast(intrinsic)
                    focalLength = scaleOffset.getFocalLength()
                    sensorWidth = scaleOffset.sensorWidth()
                    focal_x_pix = focalLength * float(scaleOffset.w()) / sensorWidth
                    fov_x_deg = 2 * 180 * np.arctan(sensorWidth / ( 2 *focalLength)) / np.pi

                    scale = scaleOffset.getScale()
                    pp = scaleOffset.getPrincipalPoint()
                    intrinsics = np.array([[scale[0][0], 0, pp[0][0]], [0, scale[1][0], pp[1][0]], [0.0, 0.0, 1.0]])

                    extrinsics = np.array([])
                    if dataAV.existsPose(v):
                        poseId = v.getPoseId()
                        pose = dataAV.getAbsolutePose(poseId)
                        poseTransform = pose.getTransform()
                        rotation = poseTransform.rotation()
                        translation = poseTransform.translation()
                        extrinsics = np.array([
                            np.append(rotation[0], translation[0][0]),
                            np.append(rotation[1], translation[1][0]),
                            np.append(rotation[2], translation[2][0]),
                            np.array([0.0, 0.0, 0.0, 1.0])])
                        center = poseTransform.center()
                        CArr = [center[0][0], center[1][0], center[2][0]]
                        m = np.transpose(rotation) @ np.linalg.inv(intrinsics)
                        iCamArr = [m[0][0],m[0][1],m[0][2],m[1][0],m[1][1],m[1][2],m[2][0],m[2][1],m[2][2]]

                    image_paths.append((Path(v.getImage().getImagePath()), focal_x_pix, intrinsics, extrinsics, fov_x_deg, CArr, iCamArr, rotation))

            image_paths.sort(key=lambda x: x[0])
    else:
        raise ValueError(f"Input path '{input_path}' is not a valid path (folder or sfmData file).")
    return image_paths
