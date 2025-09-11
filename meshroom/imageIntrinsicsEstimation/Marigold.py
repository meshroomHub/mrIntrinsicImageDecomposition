__version__ = "1.0"

from re import M
from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL

class MarigoldNodeSize(desc.MultiDynamicNodeSize):
    def computeSize(self, node):
        from pathlib import Path
        import itertools

        input_path_param = node.attribute(self._params[0])
        extension_param = node.attribute(self._params[1])

        input_path = input_path_param.value
        extension = extension_param.value
        include_suffixes = [extension.lower(), extension.upper()]

        size = 1
        if Path(input_path).is_dir():
            image_paths = list(itertools.chain(*(Path(input_path).glob(f'*.{suffix}') for suffix in include_suffixes)))
            size = len(image_paths)
        elif node.attribute(self._params[0]).isLink:
            size = node.attribute(self._params[0]).inputLink.node.size
        
        return size


class MarigoldBlockSize(desc.Parallelization):
    def getSizes(self, node):
        import math

        size = node.size
        if node.attribute('blockSize').value:
            nbBlocks = int(math.ceil(float(size) / float(node.attribute('blockSize').value)))
            return node.attribute('blockSize').value, size, nbBlocks
        else:
            return size, size, 1


class Marigold(desc.Node):
    category = "Image Intrinsics"
    documentation = """This node computes depth, normal, albedo, shading and material from a monocular image using 4 Marigold deep models.
                       In case a partial depth map is provided as input, a completed depth map is estimated using Marigold-dc algorithm."""
    
    gpu = desc.Level.INTENSIVE

    size = MarigoldNodeSize(['inputImages', 'inputExtension'])
    parallelization = MarigoldBlockSize()

    inputs = [
        desc.File(
            name="inputImages",
            label="Input Images",
            description="Input images to process. Folder path or sfmData filepath.",
            value="",
        ),
        desc.File(
            name="inputDepthMaps",
            label="Input Depth Maps",
            description="Input partial depth maps. Folder path.",
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
            name="outputFormat",
            label="Output Extension",
            description="Output extension for depth map, normal map, albedo, shading and residual.",
            values=[".exr", ".npy"],
            value=".exr",
            exclusive=True,
        ),
        desc.BoolParam(
            name="computeDepth",
            label="Compute Depth Map",
            description="If this option is enabled, a depth map is computed.",
            value=True,
        ),
        desc.BoolParam(
            name="computeNormals",
            label="Compute Normal Map",
            description="If this option is enabled, a normal map is computed.",
            value=True,
        ),
        desc.BoolParam(
            name="computeLighting",
            label="Compute Lighting",
            description="If this option is enabled, albedo, shading and residual images are computed.",
            value=True,
        ),
        desc.BoolParam(
            name="computeAppearance",
            label="Compute Appearance",
            description="If this option is enabled, albedo and material images are computed.",
            value=True,
        ),
        desc.IntParam(
            name="denoisingStep",
            label="Denoising Steps",
            value=0,
            description="Number of denoising steps in diffusion. Set to 0 to use the predifined value for each feature.",
            range=(0, 50, 1),
        ),
        desc.IntParam(
            name="ensembleSize",
            label="Ensemble Size",
            value=0,
            description="Number of averaged predictions to get the final result. Set to 0 to use the predifined value for each feature.",
            range=(0, 50, 1),
        ),
        desc.IntParam(
            name="seedGenerator",
            label="Seed Generator",
            value=-1,
            description="Reproducibility seed. Set to negative value for randomized inference.",
        ),
        desc.IntParam(
            name="processingResolution",
            label="Processing Resolution",
            value=-1,
            description="Resolution to which the input is resized before performing estimation. `0` uses the original input, '-1' resolves the best default from the model checkpoint.",
            range=(-1, 8192, 1),
        ),
        desc.BoolParam(
            name="outputProcessingResolution",
            label="Output Processing Resolution",
            description="Setting this flag will output the result at the effective value of `processing_res`, otherwise the output will be resized to the input resolution.",
            value=False,
        ),
        desc.ChoiceParam(
            name="resamplingMethod",
            label="Resampling Method",
            description="Resampling method used to resize images and predictions.",
            values=["bilinear", "bicubic", "nearest"],
            value="bilinear",
            exclusive=True,
        ),
        desc.BoolParam(
            name="keepInputDepthName",
            label="Keep Input Depth Filename",
            description="If this option is enabled, the output depth map will be named as the sparse one.",
            value=True,
        ),
        desc.BoolParam(
            name="saveVisuImages",
            label="Save images for visualization",
            description="Save additional png images for depth and normal maps.",
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
            description="Output folder containing the computed images.",
            value="{nodeCacheFolder}",
        ),
        desc.File(
            name="NormalMap",
            label="Normal Map",
            description="Output normal map",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/normals_<FILESTEM>.exr",
            group="",
            enabled=lambda node: node.computeNormals.value and node.saveVisuImages.value
        ),
        desc.File(
            name="NormalMapColor",
            label="Colored Normal Map",
            description="Colored output normal map",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/normals_vis_<FILESTEM>.png",
            group="",
            enabled=lambda node: node.computeNormals.value and node.saveVisuImages.value
        ),
        desc.File(
            name="DepthMap",
            label="Depth Map",
            description="Output depth map",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/depth_<FILESTEM>.exr",
            group="",
            enabled=lambda node: node.computeDepth.value and node.saveVisuImages.value
        ),
        desc.File(
            name="DepthMapColor",
            label="Colored Depth Map",
            description="Colored output depth map",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/depth_vis_<FILESTEM>.png",
            group="",
            enabled=lambda node: node.computeDepth.value and node.saveVisuImages.value
        ),
        desc.File(
            name="InputSparseDepthMap",
            label="Input Sparse Depth Map",
            description="Colored input sparse depth map",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/input_depth_vis_<FILESTEM>.png",
            group="",
            enabled=lambda node: node.computeDepth.value and node.saveVisuImages.value and node.inputDepthMaps.isLink
        ),
        desc.File(
            name="AlbedoFromAppearance",
            label="Albedo From Appearance",
            description="Output albedo extrated from appearance model",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/albedo_appearance_<FILESTEM>.exr",
            group="",
            enabled=lambda node: node.computeAppearance.value and node.outputFormat.value == ".exr"
        ),
        desc.File(
            name="AlbedoFromLighting",
            label="Albedo From Lighting",
            description="Output albedo extrated from lighting model",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/albedo_lighting_<FILESTEM>.exr",
            group="",
            enabled=lambda node: node.computeLighting.value and node.outputFormat.value == ".exr"
        ),
        desc.File(
            name="MaterialFromAppearance",
            label="Material From Appearance",
            description="Output material extrated from appearance model",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/material_appearance_<FILESTEM>.exr",
            group="",
            enabled=lambda node: node.computeAppearance.value and node.outputFormat.value == ".exr"
        ),
        desc.File(
            name="ShadingFromLighting",
            label="Shading From Lighting",
            description="Output shading extrated from lighting model",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/shading_lighting_<FILESTEM>.exr",
            group="",
            enabled=lambda node: node.computeLighting.value and node.outputFormat.value == ".exr"
        ),
        desc.File(
            name="ResidualFromLighting",
            label="Residual From Lighting",
            description="Output residual extrated from lighting model",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/residual_lighting_<FILESTEM>.exr",
            group="",
            enabled=lambda node: node.computeLighting.value and node.outputFormat.value == ".exr"
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
        import torch
        import os
        from img_proc import image
        from marigold_utils import loadPipe
        import numpy as np
        from pathlib import Path
        from PIL import Image

        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            if not chunk.node.inputImages.value:
                chunk.logger.warning('No input folder given.')

            chunk_image_paths = self.image_paths[chunk.range.start:chunk.range.end]

            # computation
            chunk.logger.info(f'Starting computation on chunk {chunk.range.iteration + 1}/{chunk.range.fullSize // chunk.range.blockSize + int(chunk.range.fullSize != chunk.range.blockSize)}...')

            output_dir_path = Path(chunk.node.output.value)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            processing_res = None
            if chunk.node.processingResolution.value >= 0:
                processing_res = chunk.node.processingResolution.value
            match_input_res = not chunk.node.outputProcessingResolution.value
            resample_method = chunk.node.resamplingMethod.value

            chunk.logger.info('Common processing parameters:')
            chunk.logger.info(f'    Match Input Resolution = {match_input_res}')
            chunk.logger.info(f'    Resampling Method = {resample_method}')

            if chunk.node.computeDepth.value:

                if not chunk.node.inputDepthMaps.isLink:
                    from marigold import MarigoldDepthPipeline, MarigoldDepthOutput
                    chunk.logger.info('Depth completion mode enabled')
                    pipe: MarigoldDepthPipeline = loadPipe.loadPipe("depth")
                    denoise_steps = chunk.node.denoisingStep.value if chunk.node.denoisingStep.value > 0 else 1;
                    ensemble_size = chunk.node.ensembleSize.value if chunk.node.ensembleSize.value > 0 else 10;
                else:
                    from marigold_utils.marigold_dc import MarigoldDepthCompletionPipeline, search_partial_depth
                    pipe: MarigoldDepthCompletionPipeline = loadPipe.loadPipe("depthCompletion")
                    # if not torch.cuda.is_available():
                    #     import diffusers
                    #     chunk.logger.info("CUDA not found: Using a lightweight VAE for depth completion pipeline")
                    #     del pipeCompletion.vae
                    #     pipeCompletion.vae = diffusers.AutoencoderTiny.from_pretrained("madebyollin/taesd").to('cpu')
                    denoise_steps = chunk.node.denoisingStep.value if chunk.node.denoisingStep.value > 0 else 50;
                    ensemble_size = 1

                chunk.logger.info('Depth processing parameters:')
                chunk.logger.info(f'    Processing Resolution = {processing_res or pipe.default_processing_resolution}')
                chunk.logger.info(f'    Denoising Step(s) = {denoise_steps}')
                chunk.logger.info(f'    Ensemble Size = {ensemble_size}')

                for idx, path in enumerate(chunk_image_paths):
                    input_image, h_ori, w_ori, pixelAspectRatio, orientation = image.loadImage(str(chunk_image_paths[idx][0]), applyPAR = True)
                    input_image = Image.fromarray((255.0*input_image).astype(np.uint8))

                    input_depth = None
                    if chunk.node.inputDepthMaps.isLink:
                        input_depth = search_partial_depth(chunk.node.inputDepthMaps.value, chunk.node.outputFormat.value, path, (h_ori, w_ori, pixelAspectRatio, orientation), chunk.logger)

                    depth_pred = None

                    if input_depth is not None:

                        depth_pred = pipe(
                            image=input_image,
                            sparse_depth=input_depth[0],
                            num_inference_steps=denoise_steps,
                            processing_resolution=processing_res,
                        )

                        depth_colored = pipe.image_processor.visualize_depth(depth_pred, val_min=depth_pred.min(), val_max=depth_pred.max())[0]
                        val_max = input_depth[0].max()
                        mask_pos = input_depth[0] > 0.0
                        val_min = input_depth[0][mask_pos].min()
                        input_depth_colored = pipe.image_processor.visualize_depth(input_depth[0], val_min=val_min, val_max=val_max)[0]
                        black_img = Image.new('RGB', input_depth_colored.size, (0,0,0))
                        mask0_img = Image.fromarray(input_depth[0] == 0.0)
                        input_depth_colored.paste(black_img, (0,0), mask0_img)

                    elif not chunk.node.inputDepthMaps.isLink:
                        with torch.no_grad():
                            # Random number generator
                            if chunk.node.seedGenerator.value < 0:
                                generator = None
                            else:
                                generator = torch.Generator(device=device)
                                generator.manual_seed(chunk.node.seedGenerator.value)

                            # Perform inference
                            pipe_out: MarigoldDepthOutput = pipe(
                                input_image,
                                denoising_steps=denoise_steps,
                                ensemble_size=ensemble_size,
                                processing_res=processing_res,
                                match_input_res=match_input_res,
                                batch_size=0,
                                color_map="Spectral",
                                show_progress_bar=False,
                                resample_method=resample_method,
                                generator=generator,
                            )

                            depth_pred: np.ndarray = pipe_out.depth_np
                            depth_colored: np.ndarray = np.array(pipe_out.depth_colored)

                    if depth_pred is not None:

                        image_stem = Path(chunk_image_paths[idx][0]).stem
                        image_stem = str(image_stem)

                        if input_depth is not None and chunk.node.keepInputDepthName.value:
                            depth_file_name = Path(input_depth[1]).name
                        else:
                            depth_file_name = "depth_" + image_stem + chunk.node.outputFormat.value

                        depth_file_path = str(output_dir_path / depth_file_name)

                        if chunk.node.outputFormat.value == '.npy' or (input_depth is not None and str(Path(input_depth[1]).suffix) == '.npy'):
                            # Save as npy
                            np.save(depth_file_path, depth_pred)
                        else:
                            depth_to_save = depth_pred[:,:,np.newaxis].copy()
                            image.writeImage(depth_file_path, depth_to_save, h_ori, w_ori, orientation, pixelAspectRatio)
                            if input_depth is not None:
                                minDepth = depth_pred.min()
                                maxDepth = depth_pred.max()
                                nbDepthValues = depth_pred.shape[0] * depth_pred.shape[1] # h_ori * w_ori ????
                                image.transferAVDepthMetadata(input_depth[1], depth_file_path, minDepth, maxDepth, nbDepthValues)

                        if chunk.node.saveVisuImages.value:
                            # Save Colorize
                            depth_vis_file_name = "depth_vis_" + image_stem + ".png"
                            depth_vis_file_path = str(output_dir_path / depth_vis_file_name)
                            depth_colored.save(depth_vis_file_path)

                            if input_depth is not None:

                                input_depth_vis_file_name = "input_depth_vis_" + image_stem + ".png"
                                input_depth_vis_file_path = str(output_dir_path / input_depth_vis_file_name)
                                input_depth_colored.save(input_depth_vis_file_path)

            if chunk.node.computeNormals.value:
                from marigold import MarigoldNormalsPipeline, MarigoldNormalsOutput

                pipe: MarigoldNormalsPipeline = loadPipe.loadPipe("normals")
                denoise_steps = chunk.node.denoisingStep.value if chunk.node.denoisingStep.value > 0 else 4
                ensemble_size = chunk.node.ensembleSize.value if chunk.node.ensembleSize.value > 0 else 10

                chunk.logger.info('Normals processing parameters:')
                chunk.logger.info(f'    Processing Resolution = {processing_res or pipe.default_processing_resolution}')
                chunk.logger.info(f'    Denoising Step(s) = {denoise_steps}')
                chunk.logger.info(f'    Ensemble Size = {ensemble_size}')

                for idx, path in enumerate(chunk_image_paths):
                    with torch.no_grad():
                        input_image, h_ori, w_ori, pixelAspectRatio, orientation = image.loadImage(str(chunk_image_paths[idx][0]), applyPAR = True)
                        input_image = Image.fromarray((255.0*input_image).astype(np.uint8))

                        # Random number generator
                        if chunk.node.seedGenerator.value < 0:
                            generator = None
                        else:
                            generator = torch.Generator(device=device)
                            generator.manual_seed(chunk.node.seedGenerator.value)

                        # Perform inference
                        pipe_out: MarigoldNormalsOutput = pipe(
                            input_image,
                            denoising_steps=denoise_steps,
                            ensemble_size=ensemble_size,
                            processing_res=processing_res,
                            match_input_res=match_input_res,
                            batch_size=0,
                            show_progress_bar=False,
                            resample_method=resample_method,
                            generator=generator,
                        )

                        image_stem = Path(chunk_image_paths[idx][0]).stem
                        image_stem = str(image_stem)

                        normals_pred: np.ndarray = pipe_out.normals_np
                        normals_colored: np.ndarray = np.array(pipe_out.normals_img)

                        normals_vis_file_name = "normals_vis_" + image_stem + ".png"
                        normals_vis_file_path = str(output_dir_path / normals_vis_file_name)
                        normals_file_name = "normals_" + image_stem + chunk.node.outputFormat.value
                        normals_file_path = str(output_dir_path / normals_file_name)

                        if chunk.node.outputFormat.value == '.npy':
                            # Save as npy
                            np.save(normals_file_path, normals_pred)
                        else:
                            normals_to_save = np.transpose(normals_pred, (1, 2, 0)).copy()
                            image.writeImage(normals_file_path, normals_to_save, h_ori, w_ori, orientation, pixelAspectRatio)

                        if chunk.node.saveVisuImages.value:
                            # Save Colorize
                            normals_colored.save(normals_vis_file_path)

            if chunk.node.computeAppearance.value:
                from marigold import MarigoldIIDPipeline, MarigoldIIDOutput

                pipe: MarigoldIIDPipeline = loadPipe.loadPipe("appearance")
                denoise_steps = chunk.node.denoisingStep.value if chunk.node.denoisingStep.value > 0 else 4
                ensemble_size = chunk.node.ensembleSize.value if chunk.node.ensembleSize.value > 0 else 1

                chunk.logger.info('Appearance processing parameters:')
                chunk.logger.info(f'    Processing Resolution = {processing_res or pipe.default_processing_resolution}')
                chunk.logger.info(f'    Denoising Step(s) = {denoise_steps}')
                chunk.logger.info(f'    Ensemble Size = {ensemble_size}')

                for idx, path in enumerate(chunk_image_paths):
                    with torch.no_grad():
                        input_image, h_ori, w_ori, pixelAspectRatio, orientation = image.loadImage(str(chunk_image_paths[idx][0]), applyPAR = True)
                        input_image = Image.fromarray((255.0*input_image).astype(np.uint8))

                        # Random number generator
                        if chunk.node.seedGenerator.value < 0:
                            generator = None
                        else:
                            generator = torch.Generator(device=device)
                            generator.manual_seed(chunk.node.seedGenerator.value)

                        # Perform inference
                        pipe_out: MarigoldIIDOutput = pipe(
                            input_image,
                            denoising_steps=denoise_steps,
                            ensemble_size=ensemble_size,
                            processing_res=processing_res,
                            match_input_res=match_input_res,
                            batch_size=0,
                            show_progress_bar=False,
                            resample_method=resample_method,
                            generator=generator,
                        )

                        image_stem = Path(chunk_image_paths[idx][0]).stem
                        image_stem = str(image_stem)

                        for pred_name in pipe.target_names: #["albedo", "material"]
                            pred: np.ndarray = np.moveaxis(pipe_out[pred_name].array, 0, -1).copy()
                            pred_file_name = pred_name + "_appearance_" + image_stem + chunk.node.outputFormat.value
                            pred_file_path = str(output_dir_path / pred_file_name)
                            if chunk.node.outputFormat.value == '.npy':
                                # Save as npy
                                np.save(pred_file_path, pred)
                            else:
                                image.writeImage(pred_file_path, pred, h_ori, w_ori, orientation, pixelAspectRatio)

            if chunk.node.computeLighting.value:
                from marigold import MarigoldIIDPipeline, MarigoldIIDOutput

                pipe: MarigoldIIDPipeline = loadPipe.loadPipe("lighting")
                denoise_steps = chunk.node.denoisingStep.value if chunk.node.denoisingStep.value > 0 else 4
                ensemble_size = chunk.node.denoisingStep.value if chunk.node.denoisingStep.value > 0 else 1

                chunk.logger.info('Lighting processing parameters:')
                chunk.logger.info(f'    Processing Resolution = {processing_res or pipe.default_processing_resolution}')
                chunk.logger.info(f'    Denoising Step(s) = {denoise_steps}')
                chunk.logger.info(f'    Ensemble Size = {ensemble_size}')

                for idx, path in enumerate(chunk_image_paths):
                    with torch.no_grad():
                        input_image, h_ori, w_ori, pixelAspectRatio, orientation = image.loadImage(str(chunk_image_paths[idx][0]), applyPAR = True)
                        input_image = Image.fromarray((255.0*input_image).astype(np.uint8))

                        # Random number generator
                        if chunk.node.seedGenerator.value < 0:
                            generator = None
                        else:
                            generator = torch.Generator(device=device)
                            generator.manual_seed(chunk.node.seedGenerator.value)

                        # Perform inference
                        pipe_out: MarigoldIIDOutput = pipe(
                            input_image,
                            denoising_steps=denoise_steps,
                            ensemble_size=ensemble_size,
                            processing_res=processing_res,
                            match_input_res=match_input_res,
                            batch_size=0,
                            show_progress_bar=False,
                            resample_method=resample_method,
                            generator=generator,
                        )

                        image_stem = Path(chunk_image_paths[idx][0]).stem
                        image_stem = str(image_stem)

                        for pred_name in pipe.target_names: #["albedo", "shading", "residual"]
                            pred: np.ndarray = np.moveaxis(pipe_out[pred_name].array, 0, -1).copy()
                            pred_file_name = pred_name + "_lighting_" + image_stem + chunk.node.outputFormat.value
                            pred_file_path = str(output_dir_path / pred_file_name)
                            if chunk.node.outputFormat.value == '.npy':
                                # Save as npy
                                np.save(pred_file_path, pred)
                            else:
                                image.writeImage(pred_file_path, pred, h_ori, w_ori, orientation, pixelAspectRatio)

            chunk.logger.info('Marigold end')
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
        image_paths = [(path, "") for path in image_paths]
    elif Path(input_path).suffix.lower() in [".sfm", ".abc"]:
        if Path(input_path).exists():
            dataAV = sfmData.SfMData()
            if sfmDataIO.load(dataAV, input_path, sfmDataIO.ALL):
                views = dataAV.getViews()
                for id, v in views.items():
                    image_paths.append((Path(v.getImage().getImagePath()), str(id)))
            image_paths.sort(key=lambda item: item[0])
    else:
        raise ValueError(f"Input path '{input_path}' is not a valid path (folder or sfmData file).")
    return image_paths
