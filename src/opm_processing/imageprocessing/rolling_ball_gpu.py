import numpy as np
import cupy as cp
from skimage.restoration import ball_kernel
from tqdm import trange

# 1) Fixed RawKernel: use (ky–pad_h, kx–pad_w) offsets from the padded center
_apply_kernel_2d = cp.RawKernel(r'''
extern "C" __global__
void apply_kernel_2d(
    const float* img,
    const float* intensity_diff,
    float* out,
    int img_h,
    int img_w,
    int pad_h,
    int pad_w,
    int padded_w,
    int kernel_h,
    int kernel_w
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= img_w || y >= img_h) return;

    // index of the center pixel in the padded image
    int center_idx = (y + pad_h) * padded_w + (x + pad_w);
    float min_val = 3.40282347e+38f;  // FLT_MAX

    // loop exactly like apply_kernel: for each kernel element
    for (int ky = 0; ky < kernel_h; ++ky) {
        int rel_y = ky - pad_h;                // relative row
        int row_off = rel_y * padded_w;
        for (int kx = 0; kx < kernel_w; ++kx) {
            int rel_x = kx - pad_w;            // relative col
            int ker_idx = ky * kernel_w + kx; // flat index into Δ
            float v = img[center_idx + row_off + rel_x]
                      + intensity_diff[ker_idx];
            if (v < min_val) {
                min_val = v;
            }
        }
    }
    out[y * img_w + x] = min_val;
}
''', 'apply_kernel_2d')


def rolling_ball_gpu_exact(
    image: np.ndarray,
    *,
    radius: int = 100,
    kernel: np.ndarray = None,
    nansafe: bool = False
) -> np.ndarray:
    """
    Exact GPU port of skimage.restoration.rolling_ball via a custom CUDA kernel.

    Parameters
    ----------
    image : (H, W) ndarray
        Input 2D image.
    radius : int
        Ball radius.
    kernel : ndarray, optional
        Precomputed ball_kernel; if None, built from `radius`.
    nansafe : bool
        Preserve NaNs if True.

    Returns
    -------
    background : (H, W) ndarray
        Estimated background, same dtype as input.
    """
    image = np.asarray(image)
    img = image.astype(np.float32, copy=False)

    # 1) Prepare kernel on CPU
    if kernel is None:
        kernel_cpu = ball_kernel(radius, img.ndim)
    else:
        kernel_cpu = np.asarray(kernel, dtype=np.float32, copy=False)

    kh, kw = kernel_cpu.shape
    pad_h, pad_w = kh // 2, kw // 2
    center = float(kernel_cpu[pad_h, pad_w])

    # Δ = center_intensity - kernel
    intensity_diff = center - kernel_cpu
    intensity_diff[kernel_cpu == np.inf] = np.inf
    diff_flat = intensity_diff.ravel()

    # 2) Pad image with +∞
    img_pad = np.pad(
        img,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode='constant',
        constant_values=np.inf
    )

    # 3) Transfer to GPU
    img_pad_gpu = cp.asarray(img_pad)
    diff_gpu = cp.asarray(diff_flat)

    h, w = img.shape
    padded_w = img_pad.shape[1]
    out_flat = cp.empty(h * w, dtype=cp.float32)

    # 4) Launch kernel (arrays + Python ints)
    threads = (16, 16, 1)
    blocks = (
        (w + threads[0] - 1) // threads[0],
        (h + threads[1] - 1) // threads[1],
        1
    )
    _apply_kernel_2d(
        blocks, threads,
        (
            img_pad_gpu.ravel(),
            diff_gpu,
            out_flat,
            h,
            w,
            pad_h,
            pad_w,
            padded_w,
            kh,
            kw,
        )
    )

    # 5) Post‐process
    bg_gpu = out_flat.reshape(h, w)
    if nansafe:
        mask = cp.isnan(cp.asarray(img))
        bg_gpu[mask] = cp.nan

    return bg_gpu.get().astype(np.float32)


# ---------------------------------------------------------------------
# 6-D subtractor unchanged
# ---------------------------------------------------------------------
def subtract_background_tpczyx(
    image_tpczyx,
    *,
    radius: int = 100,
    kernel: np.ndarray = None,
    nansafe: bool = False,
) -> np.ndarray:
    """
    Subtract rolling-ball background from a 6-D uint16 tensorstore,
    using the exact GPU kernel above.
    """
    try:
        shape = image_tpczyx.shape
    except AttributeError:
        raise ValueError("Input must have a .shape attribute")

    if len(shape) != 6 or image_tpczyx.dtype != np.uint16:
        raise ValueError("Expected uint16 array with shape (T,P,C,Z,Y,X)")

    T, P, C, Z, Y, X = shape
    out = np.empty(shape, dtype=np.uint16)

    for t in trange(T, desc='T', leave=True):
        for p in trange(P, desc='P', leave=False):
            for c in trange(C, desc='C', leave=False):
                for z in trange(Z, desc='Z', leave=False):
                    slice_cpu = image_tpczyx[t, p, c, z, :, :].read().result()
                    bg = rolling_ball_gpu_exact(
                        slice_cpu,
                        radius=radius,
                        kernel=kernel,
                        nansafe=nansafe
                    )
                    diff = slice_cpu.astype(np.float32, copy=False) - bg
                    np.clip(diff, 0, np.iinfo(np.uint16).max, out=diff)
                    out[t, p, c, z, :, :] = diff.astype(np.uint16, copy=False)

    return out