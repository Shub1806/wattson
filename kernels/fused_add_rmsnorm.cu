// Fused residual-add + RMSNorm, the layer-norm step in every Llama/Qwen block:
//
//     h   = x + residual                 (new residual, stored as fp16)
//     out = h / sqrt(mean(h^2) + eps) * weight
//
// PyTorch eager runs this as ~6 separate kernels (add, square, mean, add eps,
// rsqrt, two multiplies), each reading and writing the whole tensor to GPU
// memory. Fused, each element is read once and written once: the minimum
// possible memory traffic. The kernel does almost no math per byte, so it is
// memory-bound, and memory bandwidth (T4: 320 GB/s) is its speed limit.
//
// One thread block per row (token). Two versions:
//   - scalar:     each thread loads one fp16 value (2 bytes) at a time
//   - vectorized: each thread loads a half2 (two fp16 values, 4 bytes),
//                 halving the number of memory instructions

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>

constexpr int BLOCK = 256;

// Sum a value across the 32 threads of a warp using register shuffles.
__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, offset);
    return v;
}

// Sum a value across the whole thread block; every thread gets the result.
__device__ __forceinline__ float block_sum(float v) {
    __shared__ float partial[32];
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    v = warp_sum(v);
    if (lane == 0) partial[warp] = v;
    __syncthreads();
    const int num_warps = blockDim.x / 32;
    v = (threadIdx.x < num_warps) ? partial[lane] : 0.0f;
    if (warp == 0) v = warp_sum(v);
    if (threadIdx.x == 0) partial[0] = v;
    __syncthreads();
    return partial[0];
}

__global__ void fused_add_rmsnorm_scalar(
    const half* __restrict__ x, const half* __restrict__ residual,
    const half* __restrict__ weight, half* __restrict__ out,
    half* __restrict__ new_residual, int hidden, float eps) {
    const size_t base = (size_t)blockIdx.x * hidden;

    // Pass 1: add, store the new residual, accumulate the sum of squares.
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += BLOCK) {
        const half h = __float2half(__half2float(x[base + i]) + __half2float(residual[base + i]));
        new_residual[base + i] = h;
        const float hf = __half2float(h);
        sum_sq += hf * hf;
    }
    const float inv_rms = rsqrtf(block_sum(sum_sq) / hidden + eps);

    // Pass 2: normalize. The same thread wrote these values, so they are
    // still in cache, not re-read from GPU memory.
    for (int i = threadIdx.x; i < hidden; i += BLOCK) {
        const float hf = __half2float(new_residual[base + i]);
        out[base + i] = __float2half(hf * inv_rms * __half2float(weight[i]));
    }
}

__global__ void fused_add_rmsnorm_half2(
    const half2* __restrict__ x, const half2* __restrict__ residual,
    const half2* __restrict__ weight, half2* __restrict__ out,
    half2* __restrict__ new_residual, int hidden, float eps) {
    const int pairs = hidden / 2;
    const size_t base = (size_t)blockIdx.x * pairs;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < pairs; i += BLOCK) {
        const float2 a = __half22float2(x[base + i]);
        const float2 b = __half22float2(residual[base + i]);
        const half2 h = __floats2half2_rn(a.x + b.x, a.y + b.y);
        new_residual[base + i] = h;
        const float2 hf = __half22float2(h);
        sum_sq += hf.x * hf.x + hf.y * hf.y;
    }
    const float inv_rms = rsqrtf(block_sum(sum_sq) / hidden + eps);

    for (int i = threadIdx.x; i < pairs; i += BLOCK) {
        const float2 hf = __half22float2(new_residual[base + i]);
        const float2 w = __half22float2(weight[i]);
        out[base + i] = __floats2half2_rn(hf.x * inv_rms * w.x, hf.y * inv_rms * w.y);
    }
}

// Called from Python: writes into `out` and `new_residual`.
void fused_add_rmsnorm(torch::Tensor x, torch::Tensor residual, torch::Tensor weight,
                       torch::Tensor out, torch::Tensor new_residual, double eps,
                       bool vectorized) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kHalf, "x must be a CUDA fp16 tensor");
    TORCH_CHECK(x.is_contiguous() && residual.is_contiguous(), "inputs must be contiguous");
    const int hidden = x.size(-1);
    const int rows = x.numel() / hidden;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (vectorized) {
        TORCH_CHECK(hidden % 2 == 0, "vectorized version needs an even hidden size");
        fused_add_rmsnorm_half2<<<rows, BLOCK, 0, stream>>>(
            reinterpret_cast<const half2*>(x.data_ptr<at::Half>()),
            reinterpret_cast<const half2*>(residual.data_ptr<at::Half>()),
            reinterpret_cast<const half2*>(weight.data_ptr<at::Half>()),
            reinterpret_cast<half2*>(out.data_ptr<at::Half>()),
            reinterpret_cast<half2*>(new_residual.data_ptr<at::Half>()),
            hidden, (float)eps);
    } else {
        fused_add_rmsnorm_scalar<<<rows, BLOCK, 0, stream>>>(
            reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
            reinterpret_cast<half*>(out.data_ptr<at::Half>()),
            reinterpret_cast<half*>(new_residual.data_ptr<at::Half>()),
            hidden, (float)eps);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
