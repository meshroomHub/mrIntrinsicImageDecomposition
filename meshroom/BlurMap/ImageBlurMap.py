__version__ = "1.0"

import os

from meshroom.core import desc
from meshroom.core.utils import VERBOSE_LEVEL
from pyalicevision import parallelization as avpar


def _outName(attr, suffix):
    return "{nodeCacheFolder}/" + ("<FILESTEM>" if attr.node.keepFilename.value else "<VIEW_ID>") + suffix + ".exr"


class ImageBlurMap(desc.Node):
    size = avpar.DynamicViewsSize("input")
    gpu = desc.Level.INTENSIVE
    parallelization = desc.Parallelization(blockSize=50)

    category = "Image Intrinsics"
    documentation = """
Computes per-pixel blur maps on GPU.

 * Sharpness: local variance of the Laplacian of the luminance (single frame).

When an optical flow folder (output of the RaftOF node) is provided, neighbor frames t-1 and t+1 are registered on frame t:
 * Relative Blur: 1 - sharpness(t) / max(sharpness of t and of the registered neighbors).
   0 means frame t is the sharpest one at this pixel, values close to 1 mean frame t is much blurrier than its neighbors.
   Unlike the raw sharpness, it is not fooled by textureless areas.
 * Motion Blur Length: expected motion blur streak length in pixels (flow magnitude x shutter angle / 360).
 * Motion Anisotropy: in [0,1], how much the gradients are attenuated along the motion direction compared to the
   perpendicular direction. High values confirm motion blur, defocus blur stays close to 0.

Neighbor maps are rejected where the forward/backward flows are inconsistent (occlusions), which requires the RaftOF
"Compute Backward Flow" option. Images are ordered by path, as in RaftOF.
"""

    inputs = [
        desc.File(
            name="input",
            label="Input",
            description="SfMData file input.",
            value="",
        ),
        desc.File(
            name="flowFolder",
            label="Optical Flow Folder",
            description="Output folder of a RaftOF node (<FILESTEM>.exr forward flows, <FILESTEM>_bw.exr backward flows).\n"
                        "If empty, only the sharpness map is computed.",
            value="",
        ),
        desc.IntParam(
            name="windowSize",
            label="Window Size",
            description="Size in pixels of the local window used to aggregate the measures.",
            value=15,
            range=(3, 101, 2),
        ),
        desc.FloatParam(
            name="shutterAngle",
            label="Shutter Angle",
            description="Shutter angle in degrees, used to convert the inter-frame motion into a motion blur length.",
            value=180.0,
            range=(0.0, 360.0, 1.0),
        ),
        desc.FloatParam(
            name="fbThreshold",
            label="Flow Consistency Threshold",
            description="Maximum forward/backward flow inconsistency in pixels for a neighbor pixel to be used.",
            value=1.0,
            range=(0.1, 10.0, 0.1),
            advanced=True,
        ),
        desc.FloatParam(
            name="sharpnessFloor",
            label="Sharpness Floor",
            description="Value added to the sharpness and gradient energies before computing ratios, "
                        "to avoid amplifying noise in textureless areas.",
            value=0.0001,
            range=(0.0001, 0.01, 0.0001),
            advanced=True,
        ),
        desc.BoolParam(
            name="useGpu",
            label="Use GPU",
            description="Use GPU for computation if available.",
            value=True,
            invalidate=False,
        ),
        desc.BoolParam(
            name="keepFilename",
            label="Keep Filename",
            description="Keep the filename of the inputs for the outputs.",
            value=True,
        ),
        desc.ChoiceParam(
            name="verboseLevel",
            label="Verbose Level",
            description="Verbosity level (fatal, error, warning, info, debug).",
            value="info",
            values=VERBOSE_LEVEL,
            exclusive=True,
        ),
    ]

    outputs = [
        desc.File(
            name="output",
            label="Output Folder",
            description="Output folder for the blur maps.",
            value="{nodeCacheFolder}",
        ),
        desc.File(
            name="sharpness",
            label="Sharpness",
            description="Local variance of the Laplacian.",
            semantic="image",
            value=lambda attr: _outName(attr, "_sharpness"),
        ),
        desc.File(
            name="relativeBlur",
            label="Relative Blur",
            description="Blur relative to the registered neighbor frames, in [0,1].",
            semantic="image",
            value=lambda attr: _outName(attr, "_relativeBlur"),
            enabled=lambda node: bool(node.flowFolder.value),
        ),
        desc.File(
            name="motionBlurLength",
            label="Motion Blur Length",
            description="Expected motion blur length in pixels.",
            semantic="image",
            value=lambda attr: _outName(attr, "_motionBlurLength"),
            enabled=lambda node: bool(node.flowFolder.value),
        ),
        desc.File(
            name="motionAnisotropy",
            label="Motion Anisotropy",
            description="Gradient attenuation along the motion direction, in [0,1].",
            semantic="image",
            value=lambda attr: _outName(attr, "_motionAnisotropy"),
            enabled=lambda node: bool(node.flowFolder.value),
        ),
    ]

    def processChunk(self, chunk):
        from pathlib import Path
        import numpy as np
        import torch
        from pyalicevision import sfmData, sfmDataIO
        from pyalicevision import image as avimg
        from img_proc import image, blurMap

        try:
            chunk.logManager.start(chunk.node.verboseLevel.value)
            node = chunk.node

            dataAV = sfmData.SfMData()
            if not sfmDataIO.load(dataAV, node.input.value, sfmDataIO.ALL):
                raise RuntimeError(f"Cannot load SfMData: {node.input.value}")
            # Same ordering as RaftOF, so that consecutive entries are the flow pairs.
            views = sorted((Path(v.getImage().getImagePath()), str(vId)) for vId, v in dataAV.getViews().items())

            os.makedirs(node.output.value, exist_ok=True)
            device = torch.device("cuda" if node.useGpu.value and torch.cuda.is_available() else "cpu")
            k = node.windowSize.value
            floor = node.sharpnessFloor.value
            flowFolder = node.flowFolder.value
            if flowFolder and not os.path.isdir(flowFolder):
                raise RuntimeError(f"Optical flow folder not found: {flowFolder}")
            chunk.logger.info(f"Chunk range from {chunk.range.start} to {chunk.range.last}, device: {device}")

            def loadFlow(path, pixelAspectRatio, shape):
                if not os.path.exists(path):
                    return None
                # RaftOF stores vectors computed on the oriented, PAR-corrected image, laid out in file space.
                flow = image.loadImage(path, applyPAR=False, clipLow=-float("inf"), clipHigh=float("inf"),
                                       colorSpace=avimg.EImageColorSpace_NO_CONVERSION)[0][:, :, :2].copy()
                if flow.shape[:2] != shape:
                    raise RuntimeError(f"Flow {path} size {flow.shape[:2]} does not match image size {shape}")
                flow[:, :, 1] *= pixelAspectRatio  # back to non PAR-corrected pixels
                return torch.from_numpy(flow).permute(2, 0, 1)[None].to(device)

            def loadFrame(idx):
                imgPath = views[idx][0]
                img, h, w, par, orientation = image.loadImage(str(imgPath), applyPAR=False)
                y = blurMap.luminance(img).to(device)
                frame = {"y": y, "s": blurMap.sharpness(y, k), "h": h, "w": w, "par": par, "orientation": orientation,
                         "fw": None, "bw": None}
                if flowFolder:
                    stem = imgPath.stem
                    frame["fw"] = loadFlow(os.path.join(flowFolder, stem + ".exr"), par, img.shape[:2])
                    frame["bw"] = loadFlow(os.path.join(flowFolder, stem + "_bw.exr"), par, img.shape[:2])
                return frame

            def write(frame, viewIdx, suffix, tensor):
                imgPath, viewId = views[viewIdx]
                name = (imgPath.stem if node.keepFilename.value else viewId) + suffix + ".exr"
                data = np.ascontiguousarray(tensor[0, 0].cpu().numpy()[:, :, None])
                image.writeImage(os.path.join(node.output.value, name), data,
                                 frame["h"], frame["w"], frame["orientation"], frame["par"])

            cache = {}
            with torch.no_grad():
                # Meshroom's last chunk range goes one past the last view.
                for t in range(chunk.range.start, min(chunk.range.end, len(views))):
                    needed = [i for i in (t - 1, t, t + 1) if 0 <= i < len(views)] if flowFolder else [t]
                    for i in list(cache):
                        if i not in needed:
                            del cache[i]
                    for i in needed:
                        if i not in cache:
                            cache[i] = loadFrame(i)

                    cur = cache[t]
                    chunk.logger.info(f"{t}: {views[t][0]}")
                    write(cur, t, "_sharpness", cur["s"])
                    if not flowFolder:
                        continue

                    neighbors, flows = [], []
                    if cur["fw"] is not None and t + 1 in cache:
                        nxt = cache[t + 1]
                        neighbors.append(blurMap.warpValid(nxt["s"], cur["fw"], nxt["bw"], node.fbThreshold.value))
                        flows.append(cur["fw"])
                    if cur["bw"] is not None and t - 1 in cache:
                        prv = cache[t - 1]
                        neighbors.append(blurMap.warpValid(prv["s"], cur["bw"], prv["fw"], node.fbThreshold.value))
                        flows.append(cur["bw"])
                    if not flows:
                        chunk.logger.warning(f"No optical flow found for {views[t][0]}, flow based maps set to 0.")

                    write(cur, t, "_relativeBlur", blurMap.relativeBlur(cur["s"], neighbors, floor))
                    if flows:
                        length, aniso = blurMap.motionBlur(cur["y"], flows, node.shutterAngle.value, k, floor)
                    else:
                        length = aniso = torch.zeros_like(cur["s"])
                    write(cur, t, "_motionBlurLength", length)
                    write(cur, t, "_motionAnisotropy", aniso)

        finally:
            torch.cuda.empty_cache()
            chunk.logManager.end()
