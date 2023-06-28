#include <iostream>
#include <math.h>
#include <chrono>
#include <random>

// CUDA Kernel function to add the elements of two arrays on the GPU
__global__ void add(int n, float *x, float *y)
{
  int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride){
    y[i] = y[i] + x[i];
  }
}


int main(void)
{

  int N = 1<<28; //  elements

  std::cout<<"Size: "<< N*sizeof(float)<<"\n"; // 1GB of data

  // start general timer 
  const auto start = std::chrono::steady_clock::now();

  float *hx = new float[N];
  float *hy = new float[N];


  float *gx, *gy;
  cudaMalloc((void**)&gx, N*sizeof(float));
  cudaMalloc((void**)&gy, N*sizeof(float));

  // end for malloc
  const auto mem_alloc_end = std::chrono::steady_clock::now();

   std::random_device rdx;
   std::random_device rdy;   //Will be used to obtain a seed for the random number engine
   std::mt19937 genx(rdx()); 
   std::mt19937 geny(rdy()); //Standard mersenne_twister_engine seeded with rd()
   std::uniform_int_distribution<> distrib(1, N);

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    hx[i] = distrib(genx);
    hy[i] = distrib(geny);
  }
  cudaMemcpy(gx, hx, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(gy, hy, N*sizeof(float), cudaMemcpyHostToDevice);


  // end for initialization and memcpy
  const auto data_init_end = std::chrono::steady_clock::now();

// Run kernel on 1GB elements on  on the GPU

  add<<<1, 256>>>(N, gx, gy);


  cudaDeviceSynchronize();

  const auto kernel_end = std::chrono::steady_clock::now();

  // Check for errors (all values should be 3.0f)
  // float maxError = 0.0f;
  // for (int i = 0; i < N; i++)
  //   maxError = fmax(maxError, fabs(y[i]-3.0f));
  // std::cout << "Max error: " << maxError << std::endl;
  
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
