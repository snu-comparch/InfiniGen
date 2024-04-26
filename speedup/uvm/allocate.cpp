#include <sys/types.h>
#include <cuda_runtime_api.h>

extern "C" {
  void* uvm_malloc(ssize_t size, int device, cudaStream_t stream) {
    void *ptr;
    //cudaMalloc(&ptr, size);
    cudaMallocManaged(&ptr, size);
    return ptr;
  }

  void uvm_free(void* ptr, ssize_t size, int device, cudaStream_t stream) {
    cudaFree(ptr);
  }
}
