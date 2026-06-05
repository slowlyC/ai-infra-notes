#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_fp16.h>
#include <cute/tensor.hpp>

template <int kNumElemPerThread = 8>
__global__ void vector_add_local_tile_multi_elem_per_thread_half(
    half *z, int num, const half *x, const half *y, const half a, const half b, const half c) {
  using namespace cute;

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= num / kNumElemPerThread) {
    return;
  }

  Tensor tz = make_tensor(make_gmem_ptr(z), make_shape(num));
  Tensor tx = make_tensor(make_gmem_ptr(x), make_shape(num));
  Tensor ty = make_tensor(make_gmem_ptr(y), make_shape(num));

  Tensor tzr = local_tile(tz, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));
  Tensor txr = local_tile(tx, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));
  Tensor tyr = local_tile(ty, make_shape(Int<kNumElemPerThread>{}), make_coord(idx));

  Tensor txR = make_tensor_like(txr);
  Tensor tyR = make_tensor_like(tyr);
  Tensor tzR = make_tensor_like(tzr);

  // LDG.128
  copy(txr, txR);
  copy(tyr, tyR);

  half2 a2 = {a, a};
  half2 b2 = {b, b};
  half2 c2 = {c, c};

  auto tzR2 = recast<half2>(tzR);
  auto txR2 = recast<half2>(txR);
  auto tyR2 = recast<half2>(tyR);

#pragma unroll
  for (int i = 0; i < size(tzR2); ++i) {
    // two hfma2 instructions
    tzR2(i) = txR2(i) * a2 + (tyR2(i) * b2 + c2);
  }

  auto tzRx = recast<half>(tzR2);

  // STG.128
  copy(tzRx, tzr);
}

void check_cuda(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
    exit(1);
  }
}
#define CHECK_CUDA(err) check_cuda(err, __FILE__, __LINE__)


int main() {
  const int N = 1024;
  const float a_val = 2.0f, b_val = 3.0f, c_val = 1.0f;

  size_t bytes = N * sizeof(half);
  half *h_x = (half *)malloc(bytes);
  half *h_y = (half *)malloc(bytes);
  half *h_z = (half *)malloc(bytes);

  srand(42);
  for (int i = 0; i < N; i++) {
    h_x[i] = __float2half((float)(rand() % 100) / 10.0f);
    h_y[i] = __float2half((float)(rand() % 100) / 10.0f);
  }

  half *d_x, *d_y, *d_z;
  CHECK_CUDA(cudaMalloc(&d_x, bytes));
  CHECK_CUDA(cudaMalloc(&d_y, bytes));
  CHECK_CUDA(cudaMalloc(&d_z, bytes));

  CHECK_CUDA(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

  constexpr int kNumElemPerThread = 8;
  const int block_size = 128;
  const int num_threads = N / kNumElemPerThread;
  const int grid_size = (num_threads + block_size - 1) / block_size;

  vector_add_local_tile_multi_elem_per_thread_half<kNumElemPerThread>
      <<<grid_size, block_size>>>(
          d_z, N, d_x, d_y,
          __float2half(a_val), __float2half(b_val), __float2half(c_val));
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaMemcpy(h_z, d_z, bytes, cudaMemcpyDeviceToHost));

  // z = a*x + b*y + c
  int errors = 0;
  float max_diff = 0.0f;
  for (int i = 0; i < N; i++) {
    float xf = __half2float(h_x[i]);
    float yf = __half2float(h_y[i]);
    float zf = __half2float(h_z[i]);
    float expected = a_val * xf + b_val * yf + c_val;
    float diff = fabsf(zf - expected);
    if (diff > max_diff) max_diff = diff;
    if (diff > 0.2f) {
      if (errors < 5) {
        printf("  MISMATCH [%d]: x=%.3f y=%.3f  got=%.3f expected=%.3f (diff=%.4f)\n",
               i, xf, yf, zf, expected, diff);
      }
      errors++;
    }
  }

  printf("N = %d, a = %.1f, b = %.1f, c = %.1f\n", N, a_val, b_val, c_val);
  printf("Max absolute diff: %.4f\n", max_diff);
  if (errors == 0) {
    printf("PASSED (all %d elements correct within tolerance)\n", N);
  } else {
    printf("FAILED (%d / %d elements exceed tolerance)\n", errors, N);
  }

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  free(h_x);
  free(h_y);
  free(h_z);

  return errors > 0 ? 1 : 0;
}
