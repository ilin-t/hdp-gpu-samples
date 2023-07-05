#include <iostream>
#include <math.h>
#include <chrono>
#include <random>

// Error checking for CUDA methods (not for the kernel though)
#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void SimpleSumReductionKernel (int* input, int* output){
    unsigned int i = 2*threadIdx.x;
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2){
        if (threadIdx.x % stride == 0){
            input[i] += input[i + stride]; 
            //printf("Thread %d writes %d on location %d \t", threadIdx.x, input[i], i);
        }
        __syncthreads();
    }
      
    if(threadIdx.x == 0){
    	//for(int i=0; i<blockDim.x; i++){
    		//printf("%d \t", input[i]);
    	//}
        *output = input[0];
        printf("Output: %d \n", *output);
    }
}

int main(){


	int N = 8;
	std::cout<<"Size: "<< N*sizeof(int)<<" B\n";
	
	// start general timer 
	const auto start = std::chrono::steady_clock::now();
	
	// int hx[8] = {6,4,7,5,2,3,8,1};
	int hx[8] = {1,8,3,2,5,7,4,6};
	int *hy = 0;
	
	int *gx, *gy;
	
	cudaErrorCheck( cudaMalloc((void**)&gx, N*sizeof(int)));
	cudaErrorCheck( cudaMalloc((void**)&gy, sizeof(int)));
	
	 // end for malloc
	const auto mem_alloc_end = std::chrono::steady_clock::now();

	//Will be used to obtain a seed for the random number engine
	// std::random_device rdx;

	//Standard mersenne_twister_engine seeded with rd()
	// std::mt19937 genx(rdx()); 
	// std::uniform_int_distribution<> distrib(1, 100);

	// initialize hx and hy arrays on the host
	//for (int i = 0; i < N; i++) {
	//	hx[i] = i;
	//}
	
	
	//std::cout << "hx[0] = " << hx[0] << "\n";
	
	// copy the arrays from the CPU to the GPU
	cudaMemcpy(gx, hx, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(hx, gx, sizeof(int), cudaMemcpyDeviceToHost);

	// std::cout << "gx[0] = " << gx[0] << "\n";
	//std::cout << "hx[0] = " << hx[0] << "\n";

	// end for initialization and memcpy
	const auto data_init_end = std::chrono::steady_clock::now();

	// Run kernel on 1 GiB of data on a single thread on the GPU
	SimpleSumReductionKernel<<<1, 4>>>(gx, gy);

	cudaErrorCheck( cudaPeekAtLastError() ); //debug the kernel output
	cudaErrorCheck( cudaDeviceSynchronize() );

	const auto kernel_end = std::chrono::steady_clock::now();

	// copy the resulting array back to the GPU
	cudaMemcpy(&hy, &gy, sizeof(int), cudaMemcpyDeviceToHost);
	
	//std::cout << "Result: " << gy << "\n";
	//std::cout << "Result: " << &hy << "\n";

	// Free memory
	cudaFree(gx);
	cudaFree(gy);

	

	//delete [] hx;
	delete [] hy;

	const auto free_mem = std::chrono::steady_clock::now();
	

	std::cout << "Memory allocation duration: " << std::chrono::duration_cast<std::chrono::microseconds>(mem_alloc_end-start).count()/1000<< " ms \n" \
		    << "Data initialization duration: " <<  std::chrono::duration_cast<std::chrono::microseconds>(data_init_end-mem_alloc_end).count()/1000 << " ms \n" \
		    << "Kernel duration: " << std::chrono::duration_cast<std::chrono::microseconds>(kernel_end-data_init_end).count() << " microseconds \n" \
		    << "Free memory duration: " << std::chrono::duration_cast<std::chrono::microseconds>(free_mem - kernel_end).count()/1000 << " ms \n";

	return 0;
}
