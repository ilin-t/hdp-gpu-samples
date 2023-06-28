#include <iostream>
#include <math.h>
#include <chrono>
#include <random>

// add the elements of two arrays
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
      y[i] = y[i] + x[i];
}

int main(void)
{
  int N = 1<<28; 

  std::cout << "Size: " << N*sizeof(float) << "B \n"; // 2^30 B = 1 GiB

  // start general timer 
  const auto start = std::chrono::steady_clock::now();

  float *x = new float[N];
  float *y = new float[N];

  // end for malloc
  const auto mem_alloc_end = std::chrono::steady_clock::now();
   
   //Will be used to obtain a seed for the random number engine
   std::random_device rdx;
   std::random_device rdy; 

   //Standard mersenne_twister_engine seeded with rd()
   std::mt19937 genx(rdx()); 
   std::mt19937 geny(rdy()); 
   std::uniform_int_distribution<> distrib(1, N);

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    x[i] = distrib(genx);
    y[i] = distrib(geny);
  }

  // end for initialization
  const auto data_init_end = std::chrono::steady_clock::now();

  // Run kernel on 1 GiB elements on the CPU
  add(N, x, y);

  const auto kernel_end = std::chrono::steady_clock::now();

  // Free memory
  delete [] x;
  delete [] y;

  // end for freemem
  const auto free_mem = std::chrono::steady_clock::now();

  std::cout << "Memory allocation duration: " << std::chrono::duration_cast<std::chrono::microseconds>(mem_alloc_end-start).count()/1000<< " ms \n" \
            << "Data initialization duration: " <<  std::chrono::duration_cast<std::chrono::microseconds>(data_init_end-mem_alloc_end).count()/1000 << " ms \n";
  std::cout << "Kernel duration: " << std::chrono::duration_cast<std::chrono::microseconds>(kernel_end-data_init_end).count() << " microseconds \n" \
            << "Free memory duration: " << std::chrono::duration_cast<std::chrono::microseconds>(free_mem - kernel_end).count()/1000 << " ms \n";

  return 0;
}
