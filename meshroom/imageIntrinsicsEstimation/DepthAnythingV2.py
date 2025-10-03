__version__ = "1.0"

from re import M
from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL

class DepthAnythingNodeSize(desc.MultiDynamicNodeSize):
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

class DepthAnythingBlockSize(desc.Parallelization):
    def getSizes(self, node):
        import math

        size = node.size
        if node.attribute('blockSize').value:
            nbBlocks = int(math.ceil(float(size) / float(node.attribute('blockSize').value)))
            return node.attribute('blockSize').value, size, nbBlocks
        else:
            return size, size, 1


class DepthAnythingV2(desc.Node):
    category = "Image Intrinsics"
    documentation = """This node computes depth, from a monocular image using the DepthAnythingV2 deep model."""
    
    gpu = desc.Level.INTENSIVE

    size = DepthAnythingNodeSize(['inputImages', 'inputExtension'])
    parallelization = DepthAnythingBlockSize()

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
        desc.BoolParam(
            name="halfSizeModel",
            label="Half Bit Depth Model",
            description="Use Float16 precision instead of Float32 inside the deep model for much faster inference.",
            value=False,
            advanced=True,
        ),
        desc.BoolParam(
            name="metricModel",
            label="Metric Model",
            description="Use a model with metric depth output.",
            value=True,
        ),
        desc.ChoiceParam(
            name="metricModelType",
            label="Metric Model Type",
            description="Metric model best suited for indoor or outdoor depth prediction.",
            values=["indoor", "outdoor"],
            value="indoor",
            exclusive=True,
            enabled=lambda node: node.metricModel.value,
        ),
        desc.FloatParam(
            name="maxDepth",
            label="Max Depth",
            value=20.0,
            description="Maximum of the depth",
            range=(1.0, 500.0, 1.0),
            enabled=lambda node: node.metricModel.value,
        ),
        desc.IntParam(
            name="inputModelSize",
            label="Input Model Size",
            value=518,
            description="Input size of the deep model. The higher, the more VRAM used. Default 518",
            range=(128, 2048, 1),
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
    ]

    outputs = [
        desc.File(
            name='output',
            label='Output Folder',
            description="Output folder containing the normal maps saved as exr images.",
            value="{nodeCacheFolder}",
        ),
        desc.File(
            name="DepthMap",
            label="Depth Map",
            description="Output depth map",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/depth_<FILESTEM>.exr",
            group="",
            enabled=lambda node: node.outputDepth.value,
        ),
        desc.File(
            name="DepthMapColor",
            label="Colored Depth Map",
            description="Output colored depth map",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/depth_vis_<FILESTEM>.png",
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
        if chunk.node.metricModel.value:
            from metric_depth.depth_anything_v2.dpt import DepthAnythingV2
        else:
            from depth_anything_v2.dpt import DepthAnythingV2

        from pyalicevision import image as avimg

        import torch
        from img_proc import image
        from img_proc.depth_map import colorize_depth
        import json
        import os
        import numpy as np
        from pathlib import Path

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

            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
            metric_model_types = {
                'indoor': "hypersim",
                'outdoor': "vkitti"
            }

            encoder = 'vitl'

            model_name = 'depth_anything_v2'
            if chunk.node.metricModel.value:
                model_name += f'_metric_{metric_model_types[chunk.node.metricModelType.value]}'
            model_name += f'_{encoder}'
            pretrained_path = os.getenv('DEPTHANYTHINGV2_MODELS_PATH') + '/' + model_name + '.pth'

            if chunk.node.metricModel.value:
                depth_anything = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': chunk.node.maxDepth.value})
            else:
                depth_anything = DepthAnythingV2(**model_configs[encoder])
            depth_anything.load_state_dict(torch.load(pretrained_path, map_location='cpu'))
            depth_anything = depth_anything.to(device).eval()
            
            if chunk.node.halfSizeModel.value:
                depth_anything.half()

            metadata_deep_model = {}
            metadata_deep_model["Meshroom:mrImageIntrinsicsDecomposition:DeepModelName"] = model_name
            metadata_deep_model["Meshroom:mrImageIntrinsicsDecomposition:DeepModelVersion"] = "2025.05.30"

            for idx, path in enumerate(chunk_image_paths):
                with torch.no_grad():
                    img, h_ori, w_ori, pixelAspectRatio, orientation = image.loadImage(str(chunk_image_paths[idx]), applyPAR = True)

                    img_cv2 = np.take((np.clip(img, 0.0, 1.0) * 255).astype(np.uint8), [2,1,0], axis=-1)

                    input_size = chunk.node.inputModelSize.value
                    
                    depth = depth_anything.infer_image(img_cv2, input_size)
                    if not chunk.node.metricModel.value:
                        depth = (depth - depth.min()) / (depth.max() - depth.min())

                    # Write outputs
                    outputDirPath = Path(chunk.node.output.value)
                    image_stem = Path(chunk_image_paths[idx]).stem

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
                    if chunk.node.outputDepth.value and chunk.node.saveVisuImages.value:
                        import matplotlib
                        if chunk.node.metricModel.value:
                            depth = (depth - depth.min()) / (depth.max() - depth.min())
                            cmap = matplotlib.colormaps.get_cmap('Spectral')
                        else:
                            cmap = matplotlib.colormaps.get_cmap('Spectral_r')
                        colored_depth = cmap(depth)[:, :, :3]
                        image.writeImage(vis_file_path, colored_depth, h_ori, w_ori, orientation, pixelAspectRatio, metadata_deep_model)

            chunk.logger.info('DepthAnything2 end')
        finally:
            chunk.logManager.end()

def get_image_paths_list(input_path, extension):
    from pyalicevision import sfmData
    from pyalicevision import sfmDataIO
    from pathlib import Path
    import itertools

    include_suffixes = [extension.lower(), extension.upper()]
    image_paths = []

    inPath = Path(input_path)
    if inPath.is_dir():
        image_paths = sorted(itertools.chain(*(inPath.glob(f'*.{suffix}') for suffix in include_suffixes)))
    elif inPath.suffix.lower() in [".sfm", ".abc"]:
        if inPath.exists():
            dataAV = sfmData.SfMData()
            if sfmDataIO.load(dataAV, input_path, sfmDataIO.ALL):
                views = dataAV.getViews()
                for id, v in views.items():
                    image_paths.append(Path(v.getImage().getImagePath()))
            image_paths.sort()
    else:
        raise ValueError(f"Input path '{input_path}' is not a valid path (folder or sfmData file).")
    return image_paths
