import os
import numpy as np
import OpenImageIO as oiio
from pathlib import Path
from pyalicevision import image as avimg

def find_metadata(oiio_spec, name: str, default, exact: bool = True):
    values = []
    oiio_extra_attribs = oiio_spec.extra_attribs
    for i in range(len(oiio_extra_attribs)):
        pos = oiio_extra_attribs[i].name.find(name)
        if pos == 0:
            values.insert(0, oiio_spec.getattribute(oiio_extra_attribs[i].name))
        elif pos != -1 and not exact:
            values.append(oiio_spec.getattribute(oiio_extra_attribs[i].name))
    if len(values) == 0:
        values.append(default)
    return values

def apply_orientation(oiio_image, orientation, reverse: bool = False):
    if orientation > 1:
        oiio_image_buf = oiio.ImageBuf(oiio_image)
        if orientation == 2:
            oiio_image_buf = oiio.ImageBufAlgo.flop(oiio_image_buf)
        elif orientation == 3:
            oiio_image_buf = oiio.ImageBufAlgo.rotate180(oiio_image_buf)
        elif orientation == 4:
                oiio_image_buf = oiio.ImageBufAlgo.flip(oiio_image_buf)
        elif orientation == 5:
            if reverse:
                oiio_image_buf = oiio.ImageBufAlgo.flop(oiio_image_buf)
                oiio_image_buf = oiio.ImageBufAlgo.rotate270(oiio_image_buf)
            else:
                oiio_image_buf = oiio.ImageBufAlgo.rotate90(oiio_image_buf)
                oiio_image_buf = oiio.ImageBufAlgo.flop(oiio_image_buf)
        elif orientation == 6:
            if reverse:
                oiio_image_buf = oiio.ImageBufAlgo.rotate270(oiio_image_buf)
            else:
                oiio_image_buf = oiio.ImageBufAlgo.rotate90(oiio_image_buf)
        elif orientation == 7:
            if reverse:
                 oiio_image_buf = oiio.ImageBufAlgo.flop(oiio_image_buf)
                 oiio_image_buf = oiio.ImageBufAlgo.rotate90(oiio_image_buf)
            else:
                 oiio_image_buf = oiio.ImageBufAlgo.rotate270(oiio_image_buf)
                 oiio_image_buf = oiio.ImageBufAlgo.flop(oiio_image_buf)
        elif orientation == 8:
            if reverse:
                oiio_image_buf = oiio.ImageBufAlgo.rotate90(oiio_image_buf)
            else:
                oiio_image_buf = oiio.ImageBufAlgo.rotate270(oiio_image_buf)
        oiio_image = oiio_image_buf.get_pixels(format=oiio.FLOAT)
    return oiio_image


def loadImage(imagePath: str, applyPAR: bool = False, applyOrientation: bool = True, clipLow: float = 0.0, clipHigh: float = 1.0, colorSpace = avimg.EImageColorSpace_SRGB):
    oiio_input = oiio.ImageInput.open(imagePath)
    oiio_spec = oiio_input.spec()
    oiio_input.close()

    av_image = avimg.Image_RGBfColor()
    avOptRead = avimg.ImageReadOptions(colorSpace)
    avimg.readImage(imagePath, av_image, avOptRead)
    oiio_image = av_image.getNumpyArray()

    pixelAspectRatio = oiio_spec.get_float_attribute('PixelAspectRatio', 1.0)
    orientation = int(find_metadata(oiio_spec, 'Orientation', 1)[0])
    oiio_spec.attribute('Orientation', orientation)

    if orientation > 1 and applyOrientation:
        oiio_image = apply_orientation(oiio_image, orientation)

    h,w,c = oiio_image.shape

    if pixelAspectRatio != 1.0 and applyPAR:
        oiio_image_buf = oiio.ImageBuf(oiio_image)
        nh = int(float(h) / pixelAspectRatio)
        oiio_image_buf = oiio.ImageBufAlgo.resize(oiio_image_buf, roi=oiio.ROI(0, w, 0, nh, 0, 1, 0, c+1))
        oiio_image = oiio_image_buf.get_pixels(format=oiio.FLOAT)

    oiio_image_buf = oiio.ImageBuf(oiio_image)
    oiio.ImageBufAlgo.max(oiio_image_buf, oiio_image_buf, clipLow)
    oiio.ImageBufAlgo.min(oiio_image_buf, oiio_image_buf, clipHigh)
    oiio_image = oiio_image_buf.get_pixels(format=oiio.FLOAT)

    return (oiio_image, h, w, pixelAspectRatio, orientation)

def writeImage(imagePath: str, image: np.ndarray, h_tgt: int, w_tgt: int, orientation: int = 1, pixelAspectRatio: float = 1.0, metadata = {}) -> None:
    if orientation > 1:
        image = apply_orientation(image, orientation, reverse=True)
        if orientation > 4:
            tmp = h_tgt
            h_tgt = w_tgt
            w_tgt = tmp
    h,w,c = image.shape

    if h != h_tgt or w != w_tgt:
        oiio_image_buf = oiio.ImageBuf(image)
        oiio_image_buf = oiio.ImageBufAlgo.resize(oiio_image_buf, roi=oiio.ROI(0, w_tgt, 0, h_tgt, 0, 1, 0, c+1))
        image = oiio_image_buf.get_pixels(format=oiio.FLOAT)
        w = w_tgt
        h = h_tgt

    if image.dtype == 'uint8':
        if c == 1:
            av_image = avimg.Image_uchar()
        elif c == 3:
            av_image = avimg.Image_RGBColor()
        else:
            av_image = avimg.Image_RGBAColor()
    else:
        if image.dtype != 'float32':
            image = image.astype('float32')
        if c == 1:
            av_image = avimg.Image_float()
        elif c == 3:
            av_image = avimg.Image_RGBfColor()
        else:
            av_image = avimg.Image_RGBAfColor()
    av_image.fromNumpyArray(image)

    optWrite = avimg.ImageWriteOptions()
    if Path(imagePath).suffix.lower() == ".exr":
        optWrite.toColorSpace(avimg.EImageColorSpace_NO_CONVERSION)
    else:
        optWrite.toColorSpace(avimg.EImageColorSpace_SRGB)
    compression = "zips" if Path(imagePath).suffix.lower() == ".exr" else ""
    oiio_params = avimg.oiioParams(orientation, pixelAspectRatio, compression)

    for name, value in metadata.items():
        if isinstance(name, str) and isinstance(value, str) and name != "" and value != "":
            oiio_params.add(name, value)

    avimg.writeImage(imagePath, av_image, optWrite, oiio_params.get())

def loadSequence(sequencePath: str, incolorspace: str = 'acescg', start: int = 0, stop: int = -1, verbose = False):
    rawfiles = []
    if os.path.isdir(sequencePath):
        rawfiles = sorted(os.listdir(sequencePath))
    sequence = []
    listnames = []
    for i, f in enumerate(rawfiles):
        if i >= start and (i < stop or stop == -1):
            if verbose:
                print("read {}".format(f), end=chr(13))
            sequence.append(loadImage(os.path.join(sequencePath, f), incolorspace))
            listnames.append(f)
    if verbose:
        print('\n')
    return (listnames, np.stack(sequence))

def addRectangle(image: np.ndarray, rect, color = (255, 0, 0), fill = False) -> np.ndarray:
    buf = oiio.ImageBuf(image)
    oiio.ImageBufAlgo.render_box(buf, int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3]), color, fill)
    return buf.get_pixels(format='uint8')

def addPoint(image: np.ndarray, point, color = (255, 0, 0)) -> np.ndarray:
    buf = oiio.ImageBuf(image)
    oiio.ImageBufAlgo.render_box(buf, int(point[0]) - 4, int(point[1]) - 4, int(point[0]) + 4, int(point[1]) + 4, color = color)
    oiio.ImageBufAlgo.render_line(buf, int(point[0]) - 4, int(point[1]) - 4, int(point[0]) + 4, int(point[1]) + 4, color = color)
    oiio.ImageBufAlgo.render_line(buf, int(point[0]) - 4, int(point[1]) + 4, int(point[0]) + 4, int(point[1]) - 4, color = color)
    return buf.get_pixels(format='uint8')

def addText(image: np.ndarray, text, x, y, size, color = (255, 0, 0)) -> np.ndarray:
    buf = oiio.ImageBuf(image)
    oiio.ImageBufAlgo.render_text(buf, int(x), int(y), text, int(size), "", color)
    return buf.get_pixels(format='uint8')

def transferAVDepthMetadata(srcImagePath: str, destImagePath: str, new_min_depth: float, new_max_depth: float, new_nb_depth_values: int):
    in_file = oiio.ImageInput.open(srcImagePath)
    if in_file:
        in_spec = in_file.spec()
        in_file.close()
        out_file = oiio.ImageInput.open(destImagePath)
        if out_file:
            out_spec = out_file.spec()
            out_pixels = out_file.read_image()
            out_file.close()

            out_spec.attribute("AliceVision:SensorWidth", in_spec.getattribute("AliceVision:SensorWidth"))
            out_spec.attribute("AliceVision:downscale", in_spec.getattribute("AliceVision:downscale"))
            out_spec.attribute("AliceVision:minDepth", new_min_depth)
            out_spec.attribute("AliceVision:maxDepth", new_max_depth)
            out_spec.attribute("AliceVision:nbDepthValues", new_nb_depth_values)
            out_spec.attribute("AliceVision:CArr", oiio.TypeDesc.TypeVector, in_spec.getattribute("AliceVision:CArr", oiio.TypeDesc("vec3")))
            out_spec.attribute("AliceVision:P", oiio.TypeDesc.TypeMatrix44, in_spec.getattribute("AliceVision:P", oiio.TypeDesc("MATRIX44")))
            out_spec.attribute("AliceVision:iCamArr", oiio.TypeDesc.TypeMatrix33, in_spec.getattribute("AliceVision:iCamArr", oiio.TypeDesc("MATRIX33")))

            out_updated = oiio.ImageOutput.create(destImagePath)
            if out_updated:
                out_updated.open(destImagePath, out_spec)
                out_updated.write_image(out_pixels)
                out_updated.close()

    