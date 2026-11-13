"""Per-pixel blur measures on GPU (torch). Tensors are (1, C, H, W) float32."""
import torch
import torch.nn.functional as F

_LAPLACIAN = [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]
_SOBEL_X = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]


def _conv3(x, kernel):
    k = torch.tensor(kernel, dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    return F.conv2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), k)


def box(x, k):
    p0, p1 = k // 2, (k - 1) // 2  # asymmetric for even k so the output keeps the input size
    return F.avg_pool2d(F.pad(x, (p0, p1, p0, p1), mode="reflect"), k, stride=1)


def luminance(rgb):
    """rgb: (H, W, 3) numpy/tensor -> (1, 1, H, W)."""
    rgb = torch.as_tensor(rgb)
    return (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2])[None, None]


def sharpness(y, k):
    """Local variance of the Laplacian."""
    lap = _conv3(y, _LAPLACIAN)
    return (box(lap * lap, k) - box(lap, k) ** 2).clamp_min(0)


def warp(x, flow):
    """Sample x at p + flow(p). Returns (warped, inBounds mask)."""
    _, _, h, w = x.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=x.device, dtype=x.dtype),
                            torch.arange(w, device=x.device, dtype=x.dtype), indexing="ij")
    px = xs + flow[:, 0]
    py = ys + flow[:, 1]
    grid = torch.stack((2 * px / max(w - 1, 1) - 1, 2 * py / max(h - 1, 1) - 1), dim=-1)
    warped = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)
    inside = ((px >= 0) & (px <= w - 1) & (py >= 0) & (py <= h - 1))[:, None].to(x.dtype)
    return warped, inside


def warpValid(x, flowTo, flowBack, fbThreshold):
    """Warp neighbor map x into the current frame with flowTo (current -> neighbor).
    flowBack (neighbor -> current) is used for forward/backward consistency (occlusions); may be None."""
    warped, valid = warp(x, flowTo)
    if flowBack is not None:
        back, _ = warp(flowBack, flowTo)
        err = (flowTo + back).norm(dim=1, keepdim=True)
        valid = valid * (err < fbThreshold).to(x.dtype)
    return warped, valid


def relativeBlur(sCur, neighbors, floor):
    """neighbors: list of (warped sharpness, valid mask). 0 = sharpest frame here, ->1 = blurrier than neighbors."""
    sRef = sCur
    for sNb, valid in neighbors:
        sRef = torch.maximum(sRef, sNb * valid)
    return 1 - (sCur + floor) / (sRef + floor)


def motionBlur(y, flows, shutterAngle, k, floor):
    """flows: list of available flows from the current frame (to t+1 and/or t-1).
    Returns (expected blur length in px, anisotropy in [0,1])."""
    length = torch.stack([f.norm(dim=1, keepdim=True) for f in flows]).mean(0) * shutterAngle / 360.0
    # The direction sign does not matter (squared energies), so flip the backward flow to align with the forward one.
    direction = flows[0] if len(flows) == 1 else 0.5 * (flows[0] - flows[1])
    d = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-6)
    gx = _conv3(y, _SOBEL_X)
    gy = _conv3(y, [list(r) for r in zip(*_SOBEL_X)])
    ePar = box((gx * d[:, :1] + gy * d[:, 1:]) ** 2, k)
    ePerp = box((gy * d[:, :1] - gx * d[:, 1:]) ** 2, k)
    aniso = (1 - (ePar + floor) / (ePerp + floor)).clamp(0, 1)
    aniso = aniso * (length >= 1).to(y.dtype)
    return length, aniso


if __name__ == "__main__":
    # Self-check: 3 frames shifted by 5 px, the middle one motion-blurred horizontally.
    torch.manual_seed(0)
    k = 15
    base = box(torch.rand(1, 1, 256, 296), 3)
    frames = [base[..., :, s:s + 256] for s in (0, 5, 10)]
    hk = torch.full((1, 1, 1, 9), 1 / 9.)
    blurred = F.conv2d(F.pad(frames[1], (4, 4, 0, 0), mode="reflect"), hk)
    flowFw = torch.zeros(1, 2, 256, 256); flowFw[:, 0] = -5  # content moves left by 5 px per frame
    flowBw = -flowFw
    sPrev, sMid, sNext = sharpness(frames[0], k), sharpness(blurred, k), sharpness(frames[2], k)
    nbs = [warpValid(sNext, flowFw, flowBw, 1.0), warpValid(sPrev, flowBw, flowFw, 1.0)]
    crop = (..., slice(20, -20), slice(20, -20))
    rb = relativeBlur(sMid, nbs, 1e-6)[crop].mean().item()
    rbSharp = relativeBlur(sharpness(frames[1], k), nbs, 1e-6)[crop].mean().item()
    length, aniso = motionBlur(blurred, [flowFw, flowBw], 180., k, 1e-6)
    anisoSharp = motionBlur(frames[1], [flowFw, flowBw], 180., k, 1e-6)[1][crop].mean().item()
    gauss = box(box(frames[1], 5), 5)
    anisoGauss = motionBlur(gauss, [flowFw, flowBw], 180., k, 1e-6)[1][crop].mean().item()
    print(f"relBlur blurred={rb:.3f} sharp={rbSharp:.3f} length={length.mean().item():.2f} "
          f"aniso motion={aniso[crop].mean().item():.3f} sharp={anisoSharp:.3f} gauss={anisoGauss:.3f}")
    assert rb > 0.5 and rbSharp < 0.1
    assert abs(length.mean().item() - 2.5) < 1e-4
    assert aniso[crop].mean().item() > anisoSharp + 0.3 and aniso[crop].mean().item() > anisoGauss + 0.3
    print("OK")
