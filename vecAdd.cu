#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "jhtimer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#define NUM_DATA 102400
#define MAX_BLOCKS 1024

__global__ void cudaAdd(int* _a, int* _b, int* _c) {
    int tid = threadIdx.x + blockIdx.x * MAX_BLOCKS;

    _c[tid] = _a[tid] + _b[tid];
}


int main(void) {
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int memSize = NUM_DATA * sizeof(int);

    dim3 block (MAX_BLOCKS);
    dim3 grid ((NUM_DATA  + block.x - 1) / block.x);
    
    JH_TIME timer;

    timer.set_idx(0, "total");
    timer.set_idx(1, "Computation(Kernel)");
    timer.set_idx(2, "Data Trans : Host -> Device");
    timer.set_idx(3, "Data Trans : Device -> Host");
    timer.set_idx(4, "VectorSum on Host");


    timer.start_time(0);
    a = new int[NUM_DATA]; memset(a, 0, memSize);
    b = new int[NUM_DATA]; memset(b, 0, memSize);
    c = new int[NUM_DATA]; memset(c, 0, memSize);
    
    for (int i = 0 ; i < NUM_DATA ; i++) {
        a[i] = rand() % 10;
        b[i] = rand() % 10;
    }
    
    cudaMalloc(&d_a, memSize);
    cudaMalloc(&d_b, memSize);
    cudaMalloc(&d_c, memSize);

    timer.start_time(2);
    cudaMemcpy(d_a, a, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, memSize, cudaMemcpyHostToDevice);
    timer.end_time(2);
    
    timer.start_time(1);
    cudaAdd<<<grid, block>>>(d_a, d_b, d_c);
    timer.end_time(1);

    timer.start_time(3);
    cudaMemcpy(c, d_c, memSize, cudaMemcpyDeviceToHost);
    timer.end_time(3);
    timer.end_time(0);
    

    bool result = true;
    
    timer.start_time(4);
    for (int i = 0 ; i < NUM_DATA ; i++) {
        if ((a[i] + b[i]) != c[i]) {
            printf("[%d] the result is not matched (%d %d)\n", a[i], b[i], c[i]);
            result = false;
        }
    }
    timer.end_time(4);

    if (result)  {
        printf("GPU works well\n");
    }


    timer.printAllTime();
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete[] a, delete[] b, delete[] c;
    exit(0);
}
