

// compile: nvcc -o main main.cu  -lcublas -lcudart -lcuda -lcurand
#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>

#include <iostream>
#include <stdio.h>
#include <string.h>

// Must be multiples of 16 for wmma code to work
#define MATRIX_M 16384
#define MATRIX_N 16384
#define MATRIX_K 16384

// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

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

#define curandErrCheck(stat)                         \
    {                                                \
        curandErrCheck_((stat), __FILE__, __LINE__); \
    }

void curandErrCheck_(curandStatus_t stat, const char* file, int line)
{
    if (stat != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
    }
}


__global__ void convertFp32ToFp16(half* out, float* in, int n)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
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

void bench() {
}

void run(const char * fn, const char* func){
        // load module
        char * file_name = concat(fn, ".cubin");
	CUresult cuResult;
    CUcontext cuContext;


        CUmodule module;
        CUfunction kernel;

    // Initialize CUDA context
    cuResult = cuInit(0);
    if (cuResult != CUDA_SUCCESS) {
        std::cerr << "cuInit failed with error code: " << cuResult << std::endl;
        return ;
    }

    // Create CUDA context
    cuResult = cuCtxCreate(&cuContext, 0, 0);
    if (cuResult != CUDA_SUCCESS) {
        std::cerr << "cuCtxCreate failed with error code: " << cuResult << std::endl;
        return ;
    }

        cuResult = cuModuleLoad(&module, file_name);

    if (cuResult != CUDA_SUCCESS) {
        std::cerr << "cuModuleLoad failed with error code: " << cuResult << std::endl;
        // Handle the failure appropriately
        return ;
    }
        cuResult = cuModuleGetFunction(&kernel, module, func);
    if (cuResult != CUDA_SUCCESS) {
        std::cerr << "cuModuleGetFunction failed with error code: " << cuResult << std::endl;
        // Handle the failure appropriately, maybe cleanup
        cuModuleUnload(module); // Unload the module in case of failure
        return ;
    }

        // args
        float* a_fp32;
        float* b_fp32;
        half* a_fp16;
        half* b_fp16;

        float* c;
        float* c_wmma;

        float* c_host_wmma;

        curandGenerator_t gen;

        cudaEvent_t startWMMA;
        cudaEvent_t stopWMMA;

        cudaErrCheck(cudaEventCreate(&startWMMA));
        cudaErrCheck(cudaEventCreate(&stopWMMA));

        cudaErrCheck(cudaMalloc((void**)&a_fp32, MATRIX_M * MATRIX_K * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&b_fp32, MATRIX_K * MATRIX_N * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
        cudaErrCheck(cudaMalloc((void**)&b_fp16, MATRIX_K * MATRIX_N * sizeof(half)));

        cudaErrCheck(cudaMalloc((void**)&c, MATRIX_M * MATRIX_N * sizeof(float)));
        cudaErrCheck(cudaMalloc((void**)&c_wmma, MATRIX_M * MATRIX_N * sizeof(float)));

        c_host_wmma = (float*)malloc(MATRIX_M * MATRIX_N * sizeof(float));

        curandErrCheck(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
        curandErrCheck(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

        curandErrCheck(curandGenerateUniform(gen, a_fp32, MATRIX_M * MATRIX_K));
        curandErrCheck(curandGenerateUniform(gen, b_fp32, MATRIX_K * MATRIX_N));

        // curand doesn't currently support fp16 so we generate in fp32 and convert to fp16.
        convertFp32ToFp16<<<(MATRIX_M * MATRIX_K + 255) / 256, 256>>>(a_fp16, a_fp32, MATRIX_M * MATRIX_K);
        convertFp32ToFp16<<<(MATRIX_K * MATRIX_N + 255) / 256, 256>>>(b_fp16, b_fp32, MATRIX_K * MATRIX_N);

        curandErrCheck(curandGenerateUniform(gen, c, MATRIX_M * MATRIX_N));

        curandErrCheck(curandDestroyGenerator(gen));

        cudaErrCheck(cudaMemcpy(c_wmma, c, MATRIX_M * MATRIX_N * sizeof(float), cudaMemcpyDeviceToDevice));

	// int *output;
	// cudaMalloc((void**)&output, sizeof(int)*128);


	// void * args[1] = {&output};

	// cuLaunchKernel(kernel, 1, 1, 1,
	// 		32, 1, 1,
	// 		32*1024, 0, args, 0);

	// Set arguments
	int M =  MATRIX_M;
	int N =  MATRIX_N;
	int K =  MATRIX_K;
        float alpha = 2.0f;
        float beta = 2.0f;

        printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta);

	void* args[] = { &a_fp16, &b_fp16, &c_wmma, &M, &N, &K, &alpha, &beta };

        // kernel args
        dim3 gridDim;
        dim3 blockDim;

        // blockDim.x must be a multple of warpSize
        // 128x4 means we have 16 warps and a block computes a 64x64 output tile
        blockDim.x = 128;
        blockDim.y = 4;

        gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
        gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

        printf("gridDim %d, %d\n", gridDim.x, gridDim.y);  // 256, 256
        printf("blockDim %d, %d\n", blockDim.x, blockDim.y);  // 128, 4


	// Launch the kernel
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

	// TODO benchmark
        float milliseconds = 0.0f;

    cudaErrCheck(cudaEventElapsedTime(&milliseconds, startWMMA, stopWMMA));

    std::cout << "Time taken: " << milliseconds << " milliseconds" << std::endl;

    cudaErrCheck(cudaEventDestroy(startWMMA));
    cudaErrCheck(cudaEventDestroy(stopWMMA));

}



int main(){
	run("out", "_Z12wmma_exampleP6__halfS0_Pfiiiff");
	return 0;
}
