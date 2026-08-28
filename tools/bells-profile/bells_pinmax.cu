// How much host memory will this machine pin, and does chunking help?
//
// cudaMallocHost(16.7 GB) fails on a 32 GB Windows desktop, and ggml-cuda falls back to a
// pageable buffer - which costs 1.80x, measured. The fix depends on which limit is being hit:
//
//   if a single large allocation is the problem, giving the CUDA host buffer type a max size
//   makes ggml_backend_alloc_ctx_tensors_from_buft split it, and everything works
//
//   if the total is the problem, no amount of splitting helps and the model has to be
//   partially pinned or run somewhere with more RAM
//
// So: allocate in chunks until failure, report the total reached.
//
//   nvcc -O2 -o bells_pinmax.exe bells_pinmax.cu

#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

static size_t try_chunks(size_t chunk, size_t cap) {
    std::vector<void *> held;
    size_t total = 0;

    while (total + chunk <= cap) {
        void * p = nullptr;
        if (cudaMallocHost(&p, chunk) != cudaSuccess) {
            (void) cudaGetLastError();
            break;
        }
        held.push_back(p);
        total += chunk;
    }

    for (void * p : held) {
        cudaFreeHost(p);
    }
    return total;
}

int main() {
    size_t freeb = 0, totalb = 0;
    cudaMemGetInfo(&freeb, &totalb);
    printf("GPU %.1f GB free of %.1f\n\n", freeb/1e9, totalb/1e9);

    // Can one 16.7 GB block be pinned at all? This is what llama.cpp asks for today.
    void * big = nullptr;
    const size_t want = 16740ull*1024*1024;
    const bool one_shot = cudaMallocHost(&big, want) == cudaSuccess;
    (void) cudaGetLastError();
    printf("single %.1f GB allocation: %s\n", want/1e9, one_shot ? "OK" : "FAILED");
    if (one_shot) cudaFreeHost(big);

    // And in pieces? Cap at 24 GB so a success does not exhaust the machine.
    const size_t cap = 24ull*1024*1024*1024;
    printf("\n%-12s %14s\n", "chunk", "total pinned");
    printf("-------------------------------\n");
    for (size_t mb : { 4096ull, 2048ull, 1024ull, 512ull, 256ull }) {
        const size_t chunk = mb*1024*1024;
        const size_t got   = try_chunks(chunk, cap);
        printf("%6zu MB    %8.1f GB %s\n", mb, got/1e9,
               got >= want ? " <-- enough for the 30B" : "");
    }

    printf("\nIf the chunked totals clear 16.7 GB, giving the CUDA host buffer type a\n");
    printf("max size is all that is needed. If they plateau below it, the limit is on\n");
    printf("total pinned memory and splitting cannot help.\n");
    return 0;
}
