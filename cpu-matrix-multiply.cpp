#include <iostream>
#include <math.h>
#include <random>
#include <chrono>
// #include <thrust/sort.h>


void multiply(int N, int **x, int **y, int **z){

  for(int i=0; i < N; i++){
    for(int j=0; j < N; j++){
      for(int k=0; k < N; k++){
      
          z[i][j] = z[i][j] + (x[i][k] * y[k][j]);
          printf("%d \n", z[i][j]);

      }
    }
  }

}


int main(void)
{

  const int N = 500; // 1M elements

  std::cout<<"Size: "<< N*sizeof(int)<<"\n";

  // start general timer 
  const auto start = std::chrono::steady_clock::now();

  // int **hx = new int* [N][N];
  // int **hy = new int* [N][N];
  // int **hz = new int* [N][N];

//   int x[N][N];
//   int y[N][N];
//   int z[N][N];

    int *x[], **y, **z;

//   int **gx, **gy, **gz;
//   cudaMalloc((void**)&gx, N*N*sizeof(int));
//   cudaMalloc((void**)&gy, N*N*sizeof(int));
//   cudaMalloc((void**)&gz, N*N*sizeof(int));

  // end for malloc
  const auto mem_alloc_end = std::chrono::steady_clock::now();

   std::random_device rdx;
   std::random_device rdy;   //Will be used to obtain a seed for the random number engine
   std::mt19937 genx(rdx()); 
   std::mt19937 geny(rdy()); //Standard mersenne_twister_engine seeded with rd()
   std::uniform_int_distribution<> distrib(1, N);

  // initialize x and y arrays on the host
  // for (int i = 0; i < N; i++) {
  //   hx[i] = distrib(genx);
  //   hy[i] = distrib(geny);
  //   // hy[i] = 2.0f;
  //   // std::cout << "i: " << i << ", hx: " << hx[i] << ".\n";
  // }
 for (int i = 0; i < N; i++){
   for(int j=0; j < N; j++){
     x[i][j] = distrib(genx);
     y[i][j] = distrib(geny);
     z[i][j] = 0;
  }
 } 


  // end for initialization and memcpy
  const auto data_init_end = std::chrono::steady_clock::now();


  multiply(N, x, y, z);

  const auto kernel_end = std::chrono::steady_clock::now();


   // Free memory

  // delete [] hx;
  // delete [] hy;
  // delete [] hz;

  const auto free_mem = std::chrono::steady_clock::now();

  std::cout << "Memory allocation duration: " << std::chrono::duration_cast<std::chrono::microseconds>(mem_alloc_end-start).count()/1000<< " ms \n" \
            << "Data initialization duration: " <<  std::chrono::duration_cast<std::chrono::microseconds>(data_init_end-mem_alloc_end).count()/1000 << " ms \n" \
            << "Kernel duration: " << std::chrono::duration_cast<std::chrono::microseconds>(kernel_end-data_init_end).count() << " microseconds \n" \
            << "Free memory duration: " << std::chrono::duration_cast<std::chrono::microseconds>(free_mem - kernel_end).count()/1000 << " ms \n";

  return 0;
}
