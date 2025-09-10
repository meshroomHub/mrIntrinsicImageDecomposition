__version__ = "1.0"

from re import M
from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL

class MoGeNodeSize(desc.MultiDynamicNodeSize):
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
    documentation = """This node computes depth, normal, fov and mesh from a monocular image using the MoGe deep model."""
    
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
            name="automaticFOVEstimation",
            label="Automatic FOV Estimation",
            description="If this option is enabled, the MoGe model will estimate the field of view.",
            value=True,
        ),
        desc.FloatParam(
            name="horizontalFov",
            label="Horizontal FOV",
            value=50.0,
            description="If camera parameters are known, set the horizontal field of view in degrees.",
            range=(0.0, 360.0, 1.0),
            enabled=lambda node: not node.automaticFOVEstimation.value
        ),
        desc.BoolParam(
            name="saveMesh",
            label="Save Mesh",
            description="If this option is enabled, a ply file will be saved with the estimated mesh. The color will be saved as vertex colors.",
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
            name="NormalMap",
            label="Normal Map",
            description="Output normal map",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/normals_<FILESTEM>.exr",
            group="",
        ),
        desc.File(
            name="NormalMapColor",
            label="Colored Normal Map",
            description="Output colored normal map",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/normals_vis_<FILESTEM>.png",
            group="",
        ),
        desc.File(
            name="DepthMap",
            label="Depth Map",
            description="Output depth map",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/depth_<FILESTEM>.exr",
            group="",
        ),
        desc.File(
            name="DepthMapColor",
            label="Colored Depth Map",
            description="Output colored depth map",
            semantic="image",
            value=lambda attr: "{nodeCacheFolder}/depth_vis_<FILESTEM>.png",
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
        from moge.model import MoGeModel
        from moge.utils.io import save_glb, save_ply
        from moge.utils.vis import colorize_normal
        import utils3d

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
            print("Loading MoGe model...")
            model = MoGeModel.from_pretrained(os.getenv('MOGE_MODEL_PATH')).to(device).eval()

            fov_x_ =  None if chunk.node.automaticFOVEstimation.value else chunk.node.horizontalFov.value

            for idx, path in enumerate(chunk_image_paths):
                with torch.no_grad():
                    img, h_ori, w_ori, pixelAspectRatio, orientation = image.loadImage(str(chunk_image_paths[idx]), applyPAR = True)
                    image_tensor = torch.tensor(img, dtype=torch.float32, device=device).permute(2, 0, 1)
                    
                    # safe clamp between [0,1] in case of a wrong input cs 
                    image_tensor = torch.clamp(image_tensor, 0, 1)
                    
                    output = model.infer(image_tensor, fov_x=fov_x_)
                    points = output['points'].cpu().numpy()
                    depth = output['depth'].cpu().numpy()
                    mask = output['mask'].cpu().numpy()
                    intrinsics = output['intrinsics'].cpu().numpy()

                    normals, normals_mask = utils3d.numpy.points_to_normals(points, mask=mask)
                    normals = np.nan_to_num(normals, nan=0.0, posinf=1.0, neginf=0.0)

                    # Write outputs
                    outputDirPath = Path(chunk.node.output.value)
                    image_stem = Path(chunk_image_paths[idx]).stem

                    image_stem = str(image_stem)

                    vis_file_name = "depth_vis_" + image_stem + ".png"
                    vis_file_path = str(outputDirPath / vis_file_name)
                    vis_normal_file_name = "normals_vis_" + image_stem + ".png"
                    vis_normal_file_path = str(outputDirPath / vis_normal_file_name)
                    depth_file_name = "depth_" + image_stem + ".exr"
                    depth_file_path = str(outputDirPath / depth_file_name)
                    normals_file_name = "normals_" + image_stem + ".exr"
                    normals_file_path = str(outputDirPath / normals_file_name)

                    colored_depth = colorize_depth(depth).copy()
                    depth_to_write = depth[:,:,np.newaxis]
                    colored_normals = colorize_normal(normals)

                    image.writeImage(depth_file_path, 1/depth_to_write, h_ori, w_ori, orientation, pixelAspectRatio)
                    image.writeImage(normals_file_path, normals, h_ori, w_ori, orientation, pixelAspectRatio)
                    image.writeImage(vis_file_path, colored_depth, h_ori, w_ori, orientation, pixelAspectRatio)
                    image.writeImage(vis_normal_file_path, colored_normals, h_ori, w_ori, orientation, pixelAspectRatio)

                    fov_x, fov_y = utils3d.numpy.intrinsics_to_fov(intrinsics)
                    fov_file_name = "fov_" + image_stem + ".json"
                    fov_file_path = str(outputDirPath / fov_file_name)
                    with open(fov_file_path, 'w') as f:
                        json.dump({
                            'fov_x': round(float(np.rad2deg(fov_x)), 2),
                            'fov_y': round(float(np.rad2deg(fov_y)), 2),
                        }, f)

                    threshold_meshing = 0.03
                    if chunk.node.saveMesh.value:
                        ply_file_name = "mesh_" + image_stem + ".ply"
                        ply_file_path = outputDirPath / ply_file_name

                        faces, vertices, vertex_colors, vertex_uvs = utils3d.numpy.image_mesh(
                                points,
                                img.astype(np.float32),
                                utils3d.numpy.image_uv(width=w_ori, height=int(h_ori / pixelAspectRatio)),
                                mask=mask & ~(utils3d.numpy.depth_edge(depth, rtol=threshold_meshing, mask=mask) & utils3d.numpy.normals_edge(normals, tol=5, mask=normals_mask)),
                                tri=True)
                        # When exporting the model, follow the OpenGL coordinate conventions:
                        # - world coordinate system: x right, y up, z backward.
                        # - texture coordinate system: (0, 0) for left-bottom, (1, 1) for right-top.
                        vertices, vertex_uvs = vertices * [1, -1, -1], vertex_uvs * [1, -1] + [0, 1]
                        
                        save_ply(ply_file_path, vertices, faces, vertex_colors)
            
            chunk.logger.info('MoGe end')
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
