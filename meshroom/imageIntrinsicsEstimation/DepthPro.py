__version__ = "1.0"

from re import M
from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL

class DepthProNodeSize(desc.MultiDynamicNodeSize):
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

class DepthProBlockSize(desc.Parallelization):
    def getSizes(self, node):
        import math

        size = node.size
        if node.attribute('blockSize').value:
            nbBlocks = int(math.ceil(float(size) / float(node.attribute('blockSize').value)))
            return node.attribute('blockSize').value, size, nbBlocks
        else:
            return size, size, 1


class DepthPro(desc.Node):
    category = "Image Intrinsics"
    documentation = """This node computes depth, from a monocular image using the DepthPro deep model from Apple."""
    
    gpu = desc.Level.INTENSIVE

    size = DepthProNodeSize(['inputImages', 'inputExtension'])
    parallelization = DepthProBlockSize()

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
            label="Half Size Model",
            description="Use Float16 instead of Float32 inside the deep model for much faster inference.",
            value=False,
            advanced=True,
        ),
        # desc.BoolParam(
        #     name="metricModel",
        #     label="Metric Model",
        #     description="Use a model with metric depth output.",
        #     value=True,
        # ),
        # desc.ChoiceParam(
        #     name="metricModelType",
        #     label="Metric Model Type",
        #     description="Metric model best suited for indoor or outdoor depth prediction.",
        #     values=["indoor", "outdoor"],
        #     value="indoor",
        #     exclusive=True,
        #     enabled=lambda node: node.metricModel.value,
        # ),
        # desc.FloatParam(
        #     name="maxDepth",
        #     label="Max Depth",
        #     value=20.0,
        #     description="Maximum of the depth",
        #     range=(1.0, 500.0, 1.0),
        #     enabled=lambda node: node.metricModel.value,
        # ),
        # desc.IntParam(
        #     name="inputModelSize",
        #     label="Input Model Size",
        #     value=518,
        #     description="Input size of the deep model. The higher, the more VRAM used. Default 518",
        #     range=(128, 2048, 1),
        # ),
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
        from depth_pro import create_model_and_transforms
        from depth_pro.depth_pro import DepthProConfig
        from pyalicevision import sfmData
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
            chunk.logger.info("Loading DepthPro model...")

            DEPTHPRO_MONODEPTH_CONFIG_DICT = DepthProConfig(
                patch_encoder_preset="dinov2l16_384",
                image_encoder_preset="dinov2l16_384",
                checkpoint_uri=os.getenv('DEPTHPRO_MODELS_PATH') + "/depth_pro.pt",
                decoder_features=256,
                use_fov_head=True,
                fov_encoder_preset="dinov2l16_384",
            )
            model, transform = create_model_and_transforms(
                config=DEPTHPRO_MONODEPTH_CONFIG_DICT,
                device=device,
                precision=torch.half if chunk.node.halfSizeModel.value else torch.float32,
            )
            model.eval()

            for idx, path in enumerate(chunk_image_paths):
                with torch.no_grad():
                    img, h_ori, w_ori, pixelAspectRatio, orientation = image.loadImage(str(chunk_image_paths[idx]), applyPAR = True)

                    img_metadata = avimg.readImageMetadataAsMap(str(chunk_image_paths[idx]))
                    img_info = sfmData.ImageInfo(str(chunk_image_paths[idx]), w_ori, h_ori, img_metadata)
                    f_mm = img_info.getMetadataFocalLength()

                    if f_mm > 0:
                        # Convert a focal length given in mm (35mm film equivalent) to pixels
                        f_px = f_mm * np.sqrt(w_ori**2.0 + h_ori**2.0) / np.sqrt(36**2 + 24**2)
                    else:
                        f_px = None

                    img_clip = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)

                    prediction = model.infer(transform(img_clip), f_px=f_px)
                    depth = prediction["depth"].detach().cpu().numpy().squeeze()
                    if f_px is not None:
                        chunk.logger.info(f"Focal length (from exif): {f_px:0.2f}")
                    elif prediction["focallength_px"] is not None:
                        focallength_px = prediction["focallength_px"].detach().cpu().item()
                        chunk.logger.info(f"Estimated focal length: {focallength_px}")

                    inverse_depth = 1 / depth
                    # Visualize inverse depth instead of depth, clipped to [0.1m;250m] range for better visualization.
                    max_invdepth_vizu = min(inverse_depth.max(), 1 / 0.1)
                    min_invdepth_vizu = max(1 / 250, inverse_depth.min())
                    inverse_depth_normalized = (inverse_depth - min_invdepth_vizu) / (max_invdepth_vizu - min_invdepth_vizu)

                    # Write outputs
                    outputDirPath = Path(chunk.node.output.value)
                    image_stem = Path(chunk_image_paths[idx]).stem

                    image_stem = str(image_stem)

                    vis_file_name = "depth_vis_" + image_stem + ".png"
                    vis_file_path = str(outputDirPath / vis_file_name)
                    depth_file_name = "depth_" + image_stem + ".exr"
                    depth_file_path = str(outputDirPath / depth_file_name)

                    if chunk.node.outputDepth.value:
                        depth_to_write = depth[:,:,np.newaxis]
                        image.writeImage(depth_file_path, depth_to_write, h_ori, w_ori, orientation, pixelAspectRatio)
                    if chunk.node.outputDepth.value and chunk.node.saveVisuImages.value:
                        from matplotlib import pyplot as plt
                        cmap = plt.get_cmap("turbo")
                        colored_depth = cmap(inverse_depth_normalized)[..., :3]
                        image.writeImage(vis_file_path, colored_depth, h_ori, w_ori, orientation, pixelAspectRatio)

            chunk.logger.info('DepthPro end')
        finally:
            chunk.logManager.end()

def get_image_paths_list(input_path, extension):
    from pyalicevision import sfmData
    from pyalicevision import sfmDataIO
    from pathlib import Path
    import itertools

    include_suffixes = [extension.lower(), extension.upper()]
    image_paths = []

    if Path(input_path).is_dir():
        image_paths = sorted(itertools.chain(*(Path(input_path).glob(f'*.{suffix}') for suffix in include_suffixes)))
    elif Path(input_path).suffix.lower() in [".sfm", ".abc"]:
        if Path(input_path).exists():
            dataAV = sfmData.SfMData()
            if sfmDataIO.load(dataAV, input_path, sfmDataIO.ALL):
                views = dataAV.getViews()
                for id, v in views.items():
                    image_paths.append(Path(v.getImage().getImagePath()))
            image_paths.sort()
    else:
        raise ValueError(f"Input path '{input_path}' is not a valid path (folder or sfmData file).")
    return image_paths
