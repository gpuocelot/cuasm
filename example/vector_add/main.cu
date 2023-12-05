

// compile: nvcc -o main main.cu  -lcublas -lcudart -lcuda -lcurand
// ncu profiler: sudo /usr/local/cuda/bin/ncu --target-processes all --devices 0   --launch-skip 1 --launch-count 10  ./main
#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>

#include <iostream>
#include <stdio.h>
#include <string.h>
#include <string>

// Define some error checking macros.
#define cudaErrCheck(stat)                         \
    {                                              \
        cudaErrCheck_((stat), __FILE__, __LINE__); \
    }
void cudaErrCheck_(cudaError_t stat, const char* file, int line)
{
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    }
}

#define cudaDriverCheck(stat)                         \
    {                                                 \
        cudaDriverCheck_((stat), __FILE__, __LINE__); \
    }
void cudaDriverCheck_(CUresult stat, const char* file, int line)
{
    if (stat != CUDA_SUCCESS) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", stat, file, line);
    }
}

char* concat(const char *s1, const char *s2)
{
    char *result = (char*)malloc(strlen(s1) + strlen(s2) + 1); // +1 for the null-terminator
    // in real code you would check for errors in malloc here
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

void run(const char * fn, const char* func, int n, int warmup, int repeats) {
    // load module
    char * file_name = concat(fn, ".cubin");
	CUresult cuResult;
    CUcontext cuContext;

    CUmodule module;
    CUfunction kernel;

    cudaDriverCheck(cuInit(0));  // Initialize CUDA context

    cudaDriverCheck(cuCtxCreate(&cuContext, 0, 0));  // Create CUDA context

    cudaDriverCheck(cuModuleLoad(&module, file_name));

    cudaDriverCheck(cuModuleGetFunction(&kernel, module, func));

    // args
    size_t bytes = n * sizeof(float);

    cudaEvent_t startWMMA;
    cudaEvent_t stopWMMA;
    cudaErrCheck(cudaEventCreate(&startWMMA));
    cudaErrCheck(cudaEventCreate(&stopWMMA));

    // Allocate memory on the host
    float *h_a, *h_b, *h_c;
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);

    // Initialize input vectors
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    // Allocate memory on the device
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy vectors from host to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    printf("Vector add workload...\n");
    printf("n: %d\n", n);

	void* args[] = { &d_a, &d_b, &d_c, &n};

    // kernel args
    dim3 gridDim;
    dim3 blockDim;

    blockDim.x = 256;
    gridDim.x = (int)ceil((float)n / blockDim.x);

    printf("gridDim %d, %d\n", gridDim.x, gridDim.y);
    printf("blockDim %d, %d\n", blockDim.x, blockDim.y);

    // Launch the kernel
    cuLaunchKernel(
        kernel,
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        0, /* Shared memory size - if using shared memory */
        NULL, /* Stream identifier - if using streams */
        args, /* Kernel arguments */
        NULL /* Extra options */
    );
    cudaDeviceSynchronize();

    // benchmark
    printf("\nbenchmark...\n");

    // Warm-up iterations
    for (int i = 0; i < warmup; ++i) {
        cuLaunchKernel(
            kernel,
            gridDim.x, gridDim.y, gridDim.z,
            blockDim.x, blockDim.y, blockDim.z,
            0, /* Shared memory size - if using shared memory */
            NULL, /* Stream identifier - if using streams */
            args, /* Kernel arguments */
            NULL /* Extra options */
        );
        cudaDeviceSynchronize();
    }

    float totalMilliseconds = 0;
    for (int i = 0; i < repeats; ++i) {
        cudaErrCheck(cudaEventRecord(startWMMA));
        cuLaunchKernel(
            kernel,
            gridDim.x, gridDim.y, gridDim.z,
            blockDim.x, blockDim.y, blockDim.z,
            0, /* Shared memory size - if using shared memory */
            NULL, /* Stream identifier - if using streams */
            args, /* Kernel arguments */
            NULL /* Extra options */
        );
        cudaErrCheck(cudaEventRecord(stopWMMA));
        cudaErrCheck(cudaEventSynchronize(stopWMMA));

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, startWMMA, stopWMMA);
        totalMilliseconds += milliseconds;
    }
    printf("warmup %d; repeats %d\n", warmup, repeats);
    printf("average runtime %fms\n", totalMilliseconds / repeats);

    cudaErrCheck(cudaEventDestroy(startWMMA));
    cudaErrCheck(cudaEventDestroy(stopWMMA));
}


int main(int argc, char* argv[]) {
    std::string filename = "out"; // Default filename
    int n = 1000000;  // default 1M

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--fn") {
            if (i + 1 < argc) {
                filename = argv[i + 1]; // Get the filename argument
                i++; // Skip the filename argument
            } else {
                std::cerr << "Error: --fn requires an argument." << std::endl;
                return 1;
            }
        } else if (std::string(argv[i]) == "--n") {
            if (i + 1 < argc) {
                n = std::atoi(argv[i + 1]); // Get the value for n
                i++; // Skip the n argument
            } else {
                std::cerr << "Error: --n requires an argument." << std::endl;
                return 1;
            }
        }
    }

    // Use the filename argument
    std::cout << "Filename: " << filename << ".cubin" << std::endl;

    // Convert filename to char*
    const char* filenameChar = filename.c_str();

 	run(filenameChar, "_Z9vectorAddPfS_S_i", n, 5, 50);
}
