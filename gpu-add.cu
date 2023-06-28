#include <iostream>
#include <math.h>
#include <chrono>
#include <random>

// CUDA Kernel function to add the elements of two arrays on the GPU
__global__ void add(int n, float *gx, float *gy)
{
  for (int i = 0; i < n; i++)
      gy[i] = gy[i] + gx[i];
}

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


int main(void)
{

  int N = 1<<28; 

  std::cout<<"Size: "<< N*sizeof(float)<<" B\n"; // 2^30B = 1 GiB 

  // start general timer 
  const auto start = std::chrono::steady_clock::now();

  float *hx = new float[N];
  float *hy = new float[N];


  float *gx, *gy;
  cudaErrorCheck( cudaMalloc((void**)&gx, N*sizeof(float)));
  cudaErrorCheck( cudaMalloc((void**)&gy, N*sizeof(float)));


  // end for malloc
  const auto mem_alloc_end = std::chrono::steady_clock::now();

   //Will be used to obtain a seed for the random number engine
   std::random_device rdx;
   std::random_device rdy;  

   //Standard mersenne_twister_engine seeded with rd()
   std::mt19937 genx(rdx()); 
   std::mt19937 geny(rdy()); 
   std::uniform_int_distribution<> distrib(1, N);

  // initialize hx and hy arrays on the host
  for (int i = 0; i < N; i++) {
    hx[i] = distrib(genx);
    hy[i] = distrib(geny);
  }
  
  // copy the arrays from the CPU to the GPU
  cudaMemcpy(gx, hx, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gy, hy, N*sizeof(float), cudaMemcpyHostToDevice);


  // end for initialization and memcpy
  const auto data_init_end = std::chrono::steady_clock::now();

  // Run kernel on 1 GiB of data on a single thread on the GPU
  add<<<1, 1>>>(N, gx, gy);

  cudaErrorCheck( cudaPeekAtLastError() ); //debug the kernel output
  cudaErrorCheck( cudaDeviceSynchronize() );

  const auto kernel_end = std::chrono::steady_clock::now();

  // copy the resulting array back to the GPU
  cudaMemcpy(hy, gy, N*sizeof(float), cudaMemcpyDeviceToHost);


   // Free memory
  cudaFree(gx);
  cudaFree(gy);

  delete [] hx;
  delete [] hy;

  const auto free_mem = std::chrono::steady_clock::now();

  std::cout << "Memory allocation duration: " << std::chrono::duration_cast<std::chrono::microseconds>(mem_alloc_end-start).count()/1000<< " ms \n" \
            << "Data initialization duration: " <<  std::chrono::duration_cast<std::chrono::microseconds>(data_init_end-mem_alloc_end).count()/1000 << " ms \n" \
            << "Kernel duration: " << std::chrono::duration_cast<std::chrono::microseconds>(kernel_end-data_init_end).count() << " microseconds \n" \
            << "Free memory duration: " << std::chrono::duration_cast<std::chrono::microseconds>(free_mem - kernel_end).count()/1000 << " ms \n";

  return 0;
}
