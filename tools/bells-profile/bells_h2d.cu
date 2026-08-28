// Is a pinned source actually faster for host-to-device copies on this machine?
//
// The whole "pin the model's pages" argument rests on it. Measured inside llama.cpp the answer
// was buried under allocation-path questions (does -ot actually place the experts in CUDA_Host?)
// and run-to-run noise larger than the effect. So this tests the mechanism on its own: same
// bytes, same block size, same stream, only the source memory differs.
//
//   nvcc -O2 -o bells_h2d.exe bells_h2d.cu && ./bells_h2d.exe
//
// Blocks are the expert sizes that matter. `sync` reports wall time around the copy including
// any host stall, which is what BELLS' own copy timer sees; a pageable copy blocks the caller
// for the transfer, a pinned one should not.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

static double bench(size_t block, int reps, bool pinned, bool * host_blocked) {
    void * h = nullptr;
    if (pinned) {
        CK(cudaHostAlloc(&h, block, cudaHostAllocDefault));
    } else {
        h = malloc(block);
    }
    memset(h, 1, block);

    void * d = nullptr;
    CK(cudaMalloc(&d, block));

    cudaStream_t s;
    CK(cudaStreamCreate(&s));

    // warm up
    for (int i = 0; i < 8; i++) CK(cudaMemcpyAsync(d, h, block, cudaMemcpyHostToDevice, s));
    CK(cudaStreamSynchronize(s));

    // How long does the *host* sit inside cudaMemcpyAsync? For pageable memory the driver
    // stages through its own pinned buffer and blocks; for pinned memory it should return
    // almost immediately and let the copy engine work.
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));

    struct timespec t0, t1;
    timespec_get(&t0, TIME_UTC);
    CK(cudaEventRecord(e0, s));
    for (int i = 0; i < reps; i++) {
        CK(cudaMemcpyAsync(d, h, block, cudaMemcpyHostToDevice, s));
    }
    CK(cudaEventRecord(e1, s));
    timespec_get(&t1, TIME_UTC);          // before sync: pure host-side issue time
    CK(cudaStreamSynchronize(s));

    float gpu_ms = 0.0f;
    CK(cudaEventElapsedTime(&gpu_ms, e0, e1));

    const double host_ms = (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)/1e6;
    *host_blocked = host_ms > 0.5*gpu_ms;  // did the host wait for the transfer?

    printf("  %-8s block %6.2f MB   %6.2f GB/s   host in memcpy %8.2f ms of %8.2f ms  %s\n",
           pinned ? "pinned" : "pageable",
           block/1048576.0,
           (double) block*reps/1e9/(gpu_ms/1e3),
           host_ms, gpu_ms,
           *host_blocked ? "<-- HOST BLOCKS" : "async");

    CK(cudaEventDestroy(e0));
    CK(cudaEventDestroy(e1));
    CK(cudaStreamDestroy(s));
    CK(cudaFree(d));
    if (pinned) CK(cudaFreeHost(h)); else free(h);
    return (double) block*reps/1e9/(gpu_ms/1e3);
}

int main() {
    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, 0));
    printf("%s, PCIe gen %d x%d\n\n", p.name, p.pciDomainID ? 0 : 3, 16);

    const size_t sizes[] = { 1142784, 3063808, 11849728 };   // 1.09, 2.92, 11.3 MB
    const int reps = 200;

    for (size_t b : sizes) {
        bool blk = false;
        const double pageable = bench(b, reps, false, &blk);
        const double pinned   = bench(b, reps, true,  &blk);
        printf("  -> pinned is %.2fx pageable\n\n", pinned/pageable);
    }
    return 0;
}
