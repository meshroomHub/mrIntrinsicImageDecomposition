__version__ = "1.0"

from re import M
from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL

class StableNormalNodeSize(desc.MultiDynamicNodeSize):
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


class StableNormalBlockSize(desc.Parallelization):
    def getSizes(self, node):
        import math

        size = node.size
        if node.attribute('blockSize').value:
            nbBlocks = int(math.ceil(float(size) / float(node.attribute('blockSize').value)))
            return node.attribute('blockSize').value, size, nbBlocks
        else:
            return size, size, 1


class StableNormal(desc.Node):
    category = "Image Intrinsics"
    documentation = """This node computes a normal map from a monocular image using the stableNormal deep model."""
    
    gpu = desc.Level.INTENSIVE

    size = StableNormalNodeSize(['inputImages', 'inputExtension'])
    parallelization = StableNormalBlockSize()

    inputs = [
        desc.File(
            name="inputImages",
            label="Input Images",
            description="Input images to estimate the depth from. Folder path or sfmData filepath",
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
        desc.IntParam(
            name="resolution",
            label="Resolution",
            value=1024,
            description="Size for the largest image dimension.",
            range=(256, 8192, 64),
        ),
        desc.IntParam(
            name="IterationNumber",
            label="Iteration Number",
            value=10,
            description="Number of iterations.",
            range=(1, 50, 1),
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
            name="NormalMap",
            label="Normal Map",
            description="Output normal maps",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/normal_<FILESTEM>.exr",
            group="",
        )
    ]

    def preprocess(self, node):
        extension = node.inputExtension.value
        input_path = node.inputImages.value

        image_paths = get_image_paths_list(input_path, extension)

        if len(image_paths) == 0:
            raise FileNotFoundError(f'No image files found in {input_path}')

        self.image_paths = image_paths

    def processChunk(self, chunk):
        from stablenormal.pipeline_yoso_normal import YOSONormalsPipeline
        from stablenormal.pipeline_stablenormal import StableNormalPipeline
        from stablenormal.scheduler.heuristics_ddimsampler import HEURI_DDIMScheduler

        import torch
        from img_proc import image
        import os
        import numpy as np
        from pathlib import Path
        import OpenImageIO as oiio
        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            if not chunk.node.inputImages.value:
                chunk.logger.warning('No input folder given.')

            chunk_image_paths = self.image_paths[chunk.range.start:chunk.range.end]

            DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Initialize models
            print("Loading normal estimation model...")
            yosoNormalV03_path = os.getenv('YOSONORMALV03_PATH')
            x_start_pipeline = YOSONormalsPipeline.from_pretrained(yosoNormalV03_path,
                                                                   local_file_only=True, variant="fp16",
                                                                   torch_dtype=torch.float16, t_start=0).to(DEVICE)
            
            stableNormalV01_modelpath = os.getenv('STABLENORMALV01_MODELPATH')
            pipe = StableNormalPipeline.from_pretrained(stableNormalV01_modelpath, local_file_only=True,
                                                        variant="fp16", torch_dtype=torch.float16,
                                                        scheduler=HEURI_DDIMScheduler(prediction_type='sample', 
                                                                                    beta_start=0.00085, beta_end=0.0120, 
                                                                                    beta_schedule = "scaled_linear"))

            pipe.x_start_pipeline = x_start_pipeline
            pipe.to(DEVICE)
            pipe.prior.to(DEVICE, torch.float16)
            
            try:
                import xformers
                pipe.enable_xformers_memory_efficient_attention()
            except ImportError:
                print("XFormers not available, running without memory optimizations")

            # computation
            chunk.logger.info(f'Starting computation on chunk {chunk.range.iteration + 1}/{chunk.range.fullSize // chunk.range.blockSize + int(chunk.range.fullSize != chunk.range.blockSize)}...')

            for idx, path in enumerate(chunk_image_paths):
                #if idx > 0:
                with torch.no_grad():
                    image1, h_ori, w_ori, pixelAspectRatio, orientation = image.loadImage(str(chunk_image_paths[idx]), True)

                    scale = float(chunk.node.resolution.value) / float(max(h_ori, w_ori))
                    h_tgt = h_ori * scale
                    w_tgt = w_ori * scale
                    h_tgt = int(np.round(h_tgt / 64.0)) * 64
                    w_tgt = int(np.round(w_tgt / 64.0)) * 64
                    chnb = image1.shape[2]

                    oiio_image1_buf = oiio.ImageBuf(image1)
                    oiio_image1_buf = oiio.ImageBufAlgo.resize(oiio_image1_buf, roi=oiio.ROI(0, w_tgt, 0, h_tgt, 0, 1, 0, chnb+1))
                    image1 = oiio_image1_buf.get_pixels(format=oiio.FLOAT)

                    image1 = (255.0*image1).astype(np.uint8)

                    # Generate normal map
                    pipe_out = pipe(
                        image1,
                        match_input_resolution=False,
                        processing_resolution=max(h_tgt, w_tgt),
                        num_inference_steps=chunk.node.IterationNumber.value
                    )

                    prediction = pipe_out.prediction[0].copy()
                    normalMap = (prediction.clip(-1,1) + 1) / 2

                    outputDirPath = Path(chunk.node.output.value)
                    image_stem = Path(chunk_image_paths[idx]).stem
                    of_file_name = "normal_" + image_stem + ".exr"

                    image.writeImage(str(outputDirPath / of_file_name), normalMap, h_ori, w_ori, orientation, pixelAspectRatio)
            
            chunk.logger.info('Publish end')
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
