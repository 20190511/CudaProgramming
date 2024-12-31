#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include "jhtimer.h"

#define SHARED      true

#define ROW_SIZE    32
#define K_SIZE      512
#define COL_SIZE    32

#define TYPE int

__global__ void multiArray(TYPE* a, TYPE *b, TYPE* c) 
{
    int row = threadIdx.y;
    int col = threadIdx.x;
    int index = row * blockDim.x + col;

#if SHARED
    __shared__ TYPE sA[ROW_SIZE][K_SIZE];
    __shared__ TYPE sB[K_SIZE][COL_SIZE];
 
    for (int k = 0 ; k < K_SIZE ; k++) {
        sA[row][k] = a[row * K_SIZE  + k];
        sB[k][col] = b[k * COL_SIZE + col];
    }

    __syncthreads(); //matrix load wait

    for (int k = 0 ; k < K_SIZE ; k++) {
        c[index] += sA[row][k] * sB[k][col];
    }

#else
    for (int k = 0 ; k < K_SIZE ; k++) {
        c[index] += a[row * K_SIZE  + k]  * b[k * COL_SIZE + col];
    }

#endif

}


int main(void)
{

#if SHARED
    printf("Shared Mode\n");
#else
    printf("Global Mode\n");
#endif
    JH_TIME timer;

    timer.set_idx(0, "total");
    timer.set_idx(1, "Computation(Kernel)");
    timer.set_idx(2, "Data Trans : Host -> Device");
    timer.set_idx(3, "Data Trans : Device -> Host");
    timer.set_idx(4, "VectorSum on Host");



    TYPE *a, *b, *c;
    TYPE *d_a, *d_b, *d_c;
    
    a = new TYPE[ROW_SIZE * K_SIZE * sizeof(TYPE)]; memset(a, 0, sizeof(TYPE) * K_SIZE * ROW_SIZE);
    b = new TYPE[K_SIZE * COL_SIZE * sizeof(TYPE)]; memset(b, 0, sizeof(TYPE) * K_SIZE * COL_SIZE);
    c = new TYPE[ROW_SIZE * COL_SIZE * sizeof(TYPE)]; memset(c, 0, sizeof(TYPE) * COL_SIZE * ROW_SIZE);
    
    for (int i = 0 ; i < ROW_SIZE ; i++) {
        for (int j = 0 ; j  < K_SIZE ; j++) {
            a[i*K_SIZE + j] = rand() / 7;
        }
    }

    for (int i = 0 ; i < K_SIZE ; i++) {
        for (int j = 0 ; j < COL_SIZE  ; j++) {
            b[i*COL_SIZE + j] = rand() / 7;
        }
    }


    timer.start_time(0);
    cudaMalloc(&d_a, sizeof(TYPE) * K_SIZE * ROW_SIZE);
    cudaMalloc(&d_b, sizeof(TYPE) * K_SIZE * COL_SIZE);
    cudaMalloc(&d_c, sizeof(TYPE) * COL_SIZE * ROW_SIZE);
    
    timer.start_time(2);
    cudaMemcpy(d_a, a, sizeof(TYPE)*K_SIZE*ROW_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(TYPE)*K_SIZE*COL_SIZE, cudaMemcpyHostToDevice);
    timer.end_time(2);

    dim3 blockDim(COL_SIZE, ROW_SIZE);

    timer.start_time(1);
    multiArray<<<1, blockDim>>>(d_a,d_b,d_c);
    timer.end_time(1);
    
    timer.start_time(3);
    cudaMemcpy(c, d_c, sizeof(TYPE)*COL_SIZE*ROW_SIZE, cudaMemcpyDeviceToHost);
    timer.end_time(3); 
    timer.end_time(0);

    bool check = true;
    timer.start_time(4);
    for (int i = 0 ; i < ROW_SIZE ; i++) {
        for (int j = 0 ; j < COL_SIZE ; j++) {

            TYPE ans = 0.0;
            for (int k = 0 ; k < K_SIZE ; k++) {
                ans += a[i*K_SIZE + k] * b[k*COL_SIZE + j];
            }
            if (ans != c[i*COL_SIZE + j]) {
                printf("%3f is not same (ans=%3f)\n", c[i*COL_SIZE + j], ans);
                check = false;
            }
        }
    }
    timer.end_time(4);

    if (check) {
        printf("answer is correct!!\n");
    }

    
    timer.printAllTime();
    return 0 ;
}