"""Compute structural similarity for 2D and 3D CuPy arrays."""

import math

from opm_processing.cuda import preload_cuda_libraries


preload_cuda_libraries()

import cupy as cp  # noqa: E402

# --------------------------------------------------------------------------------
# 1) Separable 1D box‐filter kernels with shared memory (unchanged)
# --------------------------------------------------------------------------------
_separable_kernels_shared = r"""
extern "C" {

__global__ void box_filter_x_shared(
    const float* __restrict__ in,
    float*       __restrict__ out,
    int W, int H, int D, int r
) {
    extern __shared__ float s[];
    const int tx = blockDim.x;
    int x = blockIdx.x * tx + threadIdx.x;
    int y = blockIdx.y;
    int z = blockIdx.z;
    int base = (z * H + y) * W;

    // load into shared memory (center + halo)
    for (int i = threadIdx.x; i < tx + 2*r; i += tx) {
        int gx = min(max(blockIdx.x*tx + i - r, 0), W-1);
        s[i] = in[base + gx];
    }
    __syncthreads();

    // compute box filter from shared buffer
    if (x < W) {
        float sum = 0.0f;
        for (int i = threadIdx.x; i < threadIdx.x + 2*r + 1; ++i) {
            sum += s[i];
        }
        out[base + x] = sum / float(2*r + 1);
    }
}

__global__ void box_filter_y_shared(
    const float* __restrict__ in,
    float*       __restrict__ out,
    int W, int H, int D, int r
) {
    extern __shared__ float s[];
    const int ty = blockDim.y;
    int x = blockIdx.x;
    int y = blockIdx.y * ty + threadIdx.y;
    int z = blockIdx.z;
    int idx = (z * H + y) * W + x;

    // load into shared memory
    for (int i = threadIdx.y; i < ty + 2*r; i += ty) {
        int gy = min(max(blockIdx.y*ty + i - r, 0), H-1);
        s[i] = in[(z * H + gy) * W + x];
    }
    __syncthreads();

    // compute
    if (y < H) {
        float sum = 0.0f;
        for (int i = threadIdx.y; i < threadIdx.y + 2*r + 1; ++i) {
            sum += s[i];
        }
        out[idx] = sum / float(2*r + 1);
    }
}

__global__ void box_filter_z_shared(
    const float* __restrict__ in,
    float*       __restrict__ out,
    int W, int H, int D, int r
) {
    extern __shared__ float s[];
    const int tz = blockDim.z;
    int x = blockIdx.x;
    int y = blockIdx.y;
    int z = blockIdx.z * tz + threadIdx.z;
    int idx = (z * H + y) * W + x;

    // load into shared memory
    for (int i = threadIdx.z; i < tz + 2*r; i += tz) {
        int gz = min(max(blockIdx.z*tz + i - r, 0), D-1);
        s[i] = in[(gz * H + y) * W + x];
    }
    __syncthreads();

    // compute
    if (z < D) {
        float sum = 0.0f;
        for (int i = threadIdx.z; i < threadIdx.z + 2*r + 1; ++i) {
            sum += s[i];
        }
        out[idx] = sum / float(2*r + 1);
    }
}

}  // extern "C"
"""

_mod_sep = cp.RawModule(code=_separable_kernels_shared, options=("-std=c++11",))
_filt_x = _mod_sep.get_function("box_filter_x_shared")
_filt_y = _mod_sep.get_function("box_filter_y_shared")
_filt_z = _mod_sep.get_function("box_filter_z_shared")


# --------------------------------------------------------------------------------
# 2) Helper: separable shared‐mem box filter on a 3D buffer
# --------------------------------------------------------------------------------
def _box_filter_3d_shared(inp: cp.ndarray, win_size: int) -> cp.ndarray:
    """Apply a separable 3D box filter using CUDA shared memory.

    Parameters
    ----------
    inp
        Input ZYX CuPy array.
    win_size
        Odd box-filter window size.

    Returns
    -------
    cupy.ndarray
        Filtered array with the same shape as the input.
    """
    D, H, W = inp.shape
    r = win_size // 2

    buf1 = cp.empty_like(inp)
    buf2 = cp.empty_like(inp)

    # choose thread‐counts to respect dims
    tx = min(64, W)
    ty = min(16, H)
    tz = min(8, D)

    gx = math.ceil(W / tx)
    gy = math.ceil(H / ty)
    gz = math.ceil(D / tz)

    # X-pass
    shared_x = (tx + 2 * r) * cp.dtype(cp.float32).itemsize
    _filt_x((gx, H, D), (tx, 1, 1), (inp, buf1, W, H, D, r), shared_mem=shared_x)

    # Y-pass
    shared_y = (ty + 2 * r) * cp.dtype(cp.float32).itemsize
    _filt_y((W, gy, D), (1, ty, 1), (buf1, buf2, W, H, D, r), shared_mem=shared_y)

    # Z-pass
    shared_z = (tz + 2 * r) * cp.dtype(cp.float32).itemsize
    _filt_z((W, H, gz), (1, 1, tz), (buf2, buf1, W, H, D, r), shared_mem=shared_z)

    return buf1


# --------------------------------------------------------------------------------
# 3) Unified SSIM wrapper for 2D or 3D “ZYX” data
# --------------------------------------------------------------------------------
def structural_similarity_cupy_sep_shared(
    img1: cp.ndarray,
    img2: cp.ndarray,
    *,
    win_size: int = 7,
    data_range: float = None,
    K1: float = 0.01,
    K2: float = 0.03,
) -> float:
    """Calculate mean SSIM for 2D or 3D float32 CuPy arrays.

    Supports 2D (Y×X) or 3D (Z×Y×X) arrays,
    using separable box filters with shared memory.
    Always assumes zyx ordering.

    Parameters
    ----------
    img1 : cp.ndarray
        Value supplied for ``img1``.
    img2 : cp.ndarray
        Value supplied for ``img2``.
    win_size : int
        Value supplied for ``win size``.
    data_range : float
        Value supplied for ``data range``.
    K1 : float
        Value supplied for ``K1``.
    K2 : float
        Value supplied for ``K2``.

    Returns
    -------
    float
        Result produced by the callable.
    """
    # promote 2D to a single‐slice 3D volume
    if img1.ndim == 2:
        img1 = img1[None, ...]
        img2 = img2[None, ...]
    elif img1.ndim != 3:
        raise ValueError("Input must be 2D or 3D")

    if img1.shape != img2.shape:
        raise ValueError("Shapes must match")

    D, H, W = img1.shape

    if data_range is None:
        data_range = float(img1.max() - img1.min())
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    # compute local means and second moments
    mu1 = _box_filter_3d_shared(img1, win_size)
    mu2 = _box_filter_3d_shared(img2, win_size)
    mu11 = _box_filter_3d_shared(img1 * img1, win_size)
    mu22 = _box_filter_3d_shared(img2 * img2, win_size)
    mu12 = _box_filter_3d_shared(img1 * img2, win_size)

    # unbiased covariances
    N = win_size ** (img1.ndim)
    factor = N / float(N - 1)
    vx = (mu11 - mu1 * mu1) * factor
    vy = (mu22 - mu2 * mu2) * factor
    vxy = (mu12 - mu1 * mu2) * factor

    # SSIM map
    num = (2 * mu1 * mu2 + C1) * (2 * vxy + C2)
    denom = (mu1 * mu1 + mu2 * mu2 + C1) * (vx + vy + C2)
    S = num / denom

    # mean over all voxels (or pixels)
    return float(cp.mean(S))
