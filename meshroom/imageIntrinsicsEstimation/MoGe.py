__version__ = "1.0"

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL

class MoGeNodeSize(desc.MultiDynamicNodeSize):
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

class MoGeBlockSize(desc.Parallelization):
    def getSizes(self, node):
        import math

        size = node.size
        if node.attribute('blockSize').value:
            nbBlocks = int(math.ceil(float(size) / float(node.attribute('blockSize').value)))
            return node.attribute('blockSize').value, size, nbBlocks
        else:
            return size, size, 1


class MoGe(desc.Node):
    category = "Image Intrinsics"
    documentation = """This node computes depth, normal, fov and mesh from a monocular image using the MoGe-2 deep model."""
    
    gpu = desc.Level.INTENSIVE

    size = MoGeNodeSize(['inputImages', 'inputExtension'])
    parallelization = MoGeBlockSize()

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
            name="automaticFoVEstimation",
            label="Automatic FoV Estimation",
            description="If this option is enabled, the MoGe model will estimate the field of view in degree.",
            value=True,
        ),
        desc.FloatParam(
            name="horizontalFoV",
            label="Horizontal FoV [deg]",
            value=50.0,
            description="If camera parameters are known, set the horizontal field of view in degree.",
            range=(0.0, 180.0, 1.0),
            enabled=lambda node: not node.automaticFoVEstimation.value
        ),
        desc.ChoiceParam(
            name="foVEstimationMode",
            label="FoV Estimation Mode",
            description="Select how field of view is estimated. If 'Full Auto' is selected, it is estimated by the deep Network.",
            values=["Full Auto", "Metadata"],
            value="Full Auto",
            exclusive=True,
            enabled=lambda node: node.automaticFoVEstimation.value
        ),
        desc.BoolParam(
            name="halfSizeModel",
            label="Half Size Model",
            description="Use Float16 instead of Float32 inside the deep model for much faster inference.",
            value=False,
            advanced=True,
        ),
        desc.IntParam(
            name="resolutionLevel",
            label="Resolution Level",
            value=9,
            description="An integer [0-9] for the resolution level for inference."
                        "Higher value means more tokens and the finer details will be captured, but inference can be slower."
                        "Defaults to 9. Note that it is irrelevant to the output size, which is always the same as the input size."
                        "`resolution_level` actually controls `num_tokens`. See `num_tokens` for more details.",
            range=(0, 9, 1),
            advanced=True,
        ),
        desc.FloatParam(
            name="threshold",
            label="Threshold",
            value=0.04,
            description="Threshold for removing edges. Defaults to 0.04. Smaller value removes more edges.",
            range=(0.001, 0.1, 0.001),
            advanced=True,
        ),
        desc.BoolParam(
            name="outputDepth",
            label="Output Depth Map",
            description="If this option is enabled, a depth map is generated.",
            value=True,
        ),
        desc.BoolParam(
            name="outputNormals",
            label="Output Normal Map",
            description="If this option is enabled, a normal map is generated.",
            value=True,
        ),
        desc.BoolParam(
            name="outputPoints",
            label="Output Points",
            description="If this option is enabled, an image of depth as vector field is generated.",
            value=False,
        ),
        desc.BoolParam(
            name="outputMask",
            label="Output Mask",
            description="If this option is enabled, a mask image is generated.",
            value=False,
        ),
        desc.BoolParam(
            name="saveVisuImages",
            label="Save images for visualization",
            description="Save additional png images for depth and normal maps.",
            value=False,
        ),
        desc.BoolParam(
            name="saveMesh",
            label="Save Mesh",
            description="If this option is enabled, the estimated mesh will be saved.",
            value=False,
        ),
        desc.ChoiceParam(
            name="meshFormat",
            label="Mesh Format",
            description="Format to save mesh. In ply, the color will be saved as vertex colors, in glb, as texture.",
            values=["ply", "glb", "both"],
            value="glb",
            exclusive=True,
            enabled=lambda node: node.saveMesh.value
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
            description="Output normal map",
            semantic="image",
            value="{nodeCacheFolder}/normals_<FILESTEM>.exr",
            enabled=lambda node: node.outputNormals.value,
        ),
        desc.File(
            name="NormalMapColor",
            label="Colored Normal Map",
            description="Output colored normal map",
            semantic="image",
            value="{nodeCacheFolder}/normals_vis_<FILESTEM>.png",
            enabled=lambda node: node.outputNormals.value and node.saveVisuImages.value,
        ),
        desc.File(
            name="DepthMap",
            label="Depth Map",
            description="Output depth map",
            semantic="image",
            value="{nodeCacheFolder}/depth_<FILESTEM>.exr",
            enabled=lambda node: node.outputDepth.value,
        ),
        desc.File(
            name="DepthMapColor",
            label="Colored Depth Map",
            description="Output colored depth map",
            semantic="image",
            value="{nodeCacheFolder}/depth_vis_<FILESTEM>.png",
            enabled=lambda node: node.outputDepth.value and node.saveVisuImages.value,
        ),
        desc.File(
            name="Mask",
            label="Mask",
            description="Edge mask",
            semantic="image",
            value="{nodeCacheFolder}/mask_<FILESTEM>.exr",
            enabled=lambda node: node.outputMask.value
        ),
        desc.File(
            name="Fov",
            label="Field Of View",
            description="Output fields of view Fov_x, Fov_y in a json file",
            value="{nodeCacheFolder}/fov_<FILESTEM>.json",
        ),
        desc.File(
            name="MeshPly",
            label="Estimated Mesh .ply",
            description="Output mesh in ply format",
            value="{nodeCacheFolder}/mesh_<FILESTEM>.ply",
            enabled=lambda node: node.saveMesh.value and node.meshFormat.value in ["both", "ply"],
        ),
        desc.File(
            name="MeshGlb",
            label="Estimated Mesh .glb",
            description="Output mesh in glb format",
            value="{nodeCacheFolder}/mesh_<FILESTEM>.glb",
            enabled=lambda node: node.saveMesh.value and node.meshFormat.value in ["both", "glb"],
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
        from moge.model import import_model_class_by_version
        from moge.utils.io import save_glb, save_ply
        from moge.utils.vis import colorize_depth, colorize_normal
        from moge.utils.geometry_numpy import depth_occlusion_edge_numpy
        import utils3d

        import torch
        from img_proc import image
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
            chunk.logger.info("Loading MoGe model...")
            DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION = {
                "v1": os.getenv('MOGE2_MODELS_PATH') + "/moge-vitl/model.pt",
                "v2": os.getenv('MOGE2_MODELS_PATH') + "/moge-2-vitl-normal/model.pt",
            }
            model_version = "v2"
            pretrained_model_name_or_path = DEFAULT_PRETRAINED_MODEL_FOR_EACH_VERSION[model_version]
            model = import_model_class_by_version(model_version).from_pretrained(pretrained_model_name_or_path).to(device).eval()

            if chunk.node.halfSizeModel.value:
                model.half()

            input_fov = chunk.node.horizontalFoV.value

            metadata_deep_model = {}
            metadata_deep_model["Meshroom:mrImageIntrinsicsDecomposition:DeepModelName"] = "MoGe-2-vitl-normal"
            metadata_deep_model["Meshroom:mrImageIntrinsicsDecomposition:DeepModelVersion"] = "2025.03.06"

            for idx, path in enumerate(chunk_image_paths):
                with torch.no_grad():
                    img, h_ori, w_ori, pixelAspectRatio, orientation = image.loadImage(str(chunk_image_paths[idx][0]), applyPAR = True)
                    image_tensor = torch.tensor(img, dtype=torch.float32, device=device).permute(2, 0, 1)

                    if chunk.node.automaticFoVEstimation.value:
                        input_fov = chunk_image_paths[idx][1] if chunk.node.foVEstimationMode.value == "Metadata" else None

                    # safe clamp between [0,1] in case of a wrong input cs 
                    image_tensor = torch.clamp(image_tensor, 0, 1)
                    
                    resolution_level = chunk.node.resolutionLevel.value
                    num_tokens = None

                    output = model.infer(image_tensor, fov_x=input_fov, resolution_level=resolution_level, num_tokens=num_tokens, use_fp16=chunk.node.halfSizeModel.value)
                    points, depth, mask, intrinsics = output['points'].cpu().numpy(), output['depth'].cpu().numpy(), output['mask'].cpu().numpy(), output['intrinsics'].cpu().numpy()
                    normals = output['normal'].cpu().numpy() if 'normal' in output else None

                    fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(intrinsics)

                    # Write outputs
                    outputDirPath = Path(chunk.node.output.value)
                    image_stem = Path(chunk_image_paths[idx][0]).stem

                    image_stem = str(image_stem)

                    vis_file_name = "depth_vis_" + image_stem + ".png"
                    vis_file_path = str(outputDirPath / vis_file_name)
                    vis_normal_file_name = "normals_vis_" + image_stem + ".png"
                    vis_normal_file_path = str(outputDirPath / vis_normal_file_name)
                    depth_file_name = "depth_" + image_stem + ".exr"
                    depth_file_path = str(outputDirPath / depth_file_name)
                    normals_file_name = "normals_" + image_stem + ".exr"
                    normals_file_path = str(outputDirPath / normals_file_name)

                    points_file_name = "points_" + image_stem + ".exr"
                    points_file_path = str(outputDirPath / points_file_name)
                    mask_file_name = "mask_" + image_stem + ".exr"
                    mask_file_path = str(outputDirPath / mask_file_name)

                    if chunk.node.outputDepth.value:
                        depth_to_write = depth[:,:,np.newaxis]
                        if chunk.node.automaticFoVEstimation.value and chunk.node.foVEstimationMode.value == "Full Auto":
                            metadata_deep_model["Meshroom:mrImageIntrinsicsDecomposition:MoGe:fov_x"] = str(180*fov_x/np.pi)
                            metadata_deep_model["Meshroom:mrImageIntrinsicsDecomposition:MoGe:fov_y"] = str(180*fov_y/np.pi)
                            metadata_deep_model["Meshroom:mrImageIntrinsicsDecomposition:MoGe:fov"] = str(180*max(fov_x, fov_y)/np.pi)
                        else:
                            metadata_deep_model["Meshroom:mrImageIntrinsicsDecomposition:Input:fov"] = str(input_fov)
                        image.writeImage(depth_file_path, depth_to_write, h_ori, w_ori, orientation, pixelAspectRatio, metadata_deep_model)
                    if chunk.node.outputNormals.value:
                        normals_to_write = normals.astype(np.float32).copy()
                        normals_to_write = normals_to_write * np.array([1, -1, -1], dtype=np.float32)
                        image.writeImage(normals_file_path, normals_to_write, h_ori, w_ori, orientation, pixelAspectRatio, metadata_deep_model)
                    if chunk.node.outputDepth.value and chunk.node.saveVisuImages.value:
                        colored_depth = colorize_depth(depth).copy()
                        image.writeImage(vis_file_path, colored_depth, h_ori, w_ori, orientation, pixelAspectRatio, metadata_deep_model)
                    if chunk.node.outputNormals.value and chunk.node.saveVisuImages.value:
                        colored_normals = colorize_normal(normals).copy()
                        image.writeImage(vis_normal_file_path, colored_normals, h_ori, w_ori, orientation, pixelAspectRatio, metadata_deep_model)
                    if chunk.node.outputPoints.value:
                        points_to_write = points.copy()
                        image.writeImage(points_file_path, points_to_write, h_ori, w_ori, orientation, pixelAspectRatio, metadata_deep_model)
                    if chunk.node.outputMask.value:
                        mask_to_write = mask.astype(np.float32).copy()
                        mask_to_write = mask_to_write[:,:,np.newaxis]
                        image.writeImage(mask_file_path, mask_to_write, h_ori, w_ori, orientation, pixelAspectRatio, metadata_deep_model)

                    fov_file_name = "fov_" + image_stem + ".json"
                    fov_file_path = str(outputDirPath / fov_file_name)
                    with open(fov_file_path, 'w') as f:
                        json.dump({
                            'fov_x': round(float(np.rad2deg(fov_x)), 2),
                            'fov_y': round(float(np.rad2deg(fov_y)), 2),
                        }, f)

                    threshold_meshing = chunk.node.threshold.value
                    if chunk.node.saveMesh.value:
                        ply_file_name = "mesh_" + image_stem + ".ply"
                        mesh_file_name = "mesh_" + image_stem + ".glb"
                        ply_file_path = outputDirPath / ply_file_name
                        mesh_file_path = outputDirPath / mesh_file_name

                        mask_cleaned = mask & ~utils3d.numpy.depth_edge(depth, rtol=threshold_meshing)
                        if normals is None:
                            faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
                                points,
                                img,
                                utils3d.numpy.image_uv(width=img.shape[1], height=img.shape[0]),
                                mask=mask_cleaned,
                                tri=True
                            )
                            vertex_normals = None
                        else:
                            faces, vertices, vertex_colors, vertex_uvs, vertex_normals = utils3d.numpy.image_mesh(
                                points,
                                img,
                                utils3d.numpy.image_uv(width=img.shape[1], height=img.shape[0]),
                                normals,
                                mask=mask_cleaned,
                                tri=True
                            )
                        # When exporting the model, follow the OpenGL coordinate conventions:
                        # - world coordinate system: x right, y up, z backward.
                        # - texture coordinate system: (0, 0) for left-bottom, (1, 1) for right-top.
                        vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]
                        if normals is not None:
                            vertex_normals = vertex_normals * [1, -1, -1]

                        if chunk.node.meshFormat.value in ["both", "glb"]:
                            save_glb(mesh_file_path, vertices, faces, vertex_uvs, (img * 255.0).astype(np.uint8), vertex_normals)

                        if chunk.node.meshFormat.value in ["both", "ply"]:
                            save_ply(ply_file_path, vertices, np.zeros((0, 3), dtype=np.int32), vertex_colors, vertex_normals)

            chunk.logger.info('MoGe2 end')
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
                    fov_x_deg = 2 * 180 * np.arctan(sensorWidth / ( 2 *focalLength)) / np.pi
                    image_paths.append((Path(v.getImage().getImagePath()), fov_x_deg))

            image_paths.sort(key=lambda x: x[0])
    else:
        raise ValueError(f"Input path '{input_path}' is not a valid path (folder or sfmData file).")
    return image_paths
