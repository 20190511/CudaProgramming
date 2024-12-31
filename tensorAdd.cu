#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_DATA 102400
#define MAX_BLOCKS 1024
#define COL_COUNT  64

// 한 블록에는 1024개의 쓰레드를 넘어갈 수 없다.


__global__ void tensorAdd(int** a, int** b, int** c) {
    int row = (gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x + ((threadIdx.x + (COL_COUNT - 1) )/ COL_COUNT );
    int col = threadIdx.x % COL_COUNT;
    
    c[row][col] = b[row][col] + a[row][col];
}


int main(void) {
    dim3 block(MAX_BLOCKS);
    int row_count = (NUM_DATA + MAX_BLOCKS - 1) / MAX_BLOCKS;
    dim3 grid(( row_count + COL_COUNT - 1) / COL_COUNT,COL_COUNT);

	printf("grid.x %d grid.y %d grid.z %d\n", grid.x, grid.y, grid.z);
	printf("block.x %d block.y %d block.z %d\n", block.x, block.y, block.z);

    int **a, **b, **c;
    int **d_a, **d_b, **d_c;

    a = new int*[(MAX_BLOCKS + COL_COUNT - 1) / COL_COUNT];
    b = new int*[(MAX_BLOCKS + COL_COUNT - 1) / COL_COUNT];
    c = new int*[(MAX_BLOCKS + COL_COUNT - 1) / COL_COUNT];

    for (int i = 0, cnt=0 ; i < (MAX_BLOCKS + COL_COUNT - 1) / COL_COUNT && cnt < NUM_DATA; i++) {
        *a = new int[COL_COUNT];
        *b = new int[COL_COUNT];
        *c = new int[COL_COUNT];

        memset(*a, 0, sizeof(int) * COL_COUNT);
        memset(*b, 0, sizeof(int) * COL_COUNT);
        memset(*c, 0, sizeof(int) * COL_COUNT);

        for (int j = 0 ; cnt < NUM_DATA ; cnt++) {
            (*a)[j] = rand() % 10;
            (*b)[j] = rand() % 10;
        }
    }
    
    
    cudaMalloc(&d_a, NUM_DATA * sizeof(int));
    cudaMalloc(&d_b, NUM_DATA * sizeof(int));
    cudaMalloc(&d_c, NUM_DATA * sizeof(int));

    for (int i = 0, cnt = 0 ; i < (NUM_DATA + COL_COUNT -1) / COL_COUNT && cnt < MAX_BLOCKS ; i++) {
        for (int j = 0 ; j < COL_COUNT && cnt < MAX_BLOCKS ; j++, cnt++) {
            c[i][j] = a[i][j] + b[i][j]; 
        }
    }
    
    cudaMemcpy(d_a, a, sizeof(int) * MAX_BLOCKS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(int) * MAX_BLOCKS, cudaMemcpyHostToDevice);

    tensorAdd<<<grid, block>>>(d_a, d_b, d_c);

    cudaMemcpy(c, d_c, sizeof(int) * MAX_BLOCKS, cudaMemcpyDeviceToHost);

    bool check = true;
    
    for (int i = 0, cnt = 0 ; i < (NUM_DATA + COL_COUNT -1) / COL_COUNT && cnt < MAX_BLOCKS ; i++) {
        for (int j = 0 ; j < COL_COUNT && cnt < MAX_BLOCKS ; j++, cnt++) {
            if ((a[i][j] + b[i][j]) != c[i][j]) {
                printf("%d + %d != c[%d][%d](%d) is not correct\n", a[i][j],b[i][j],i,j,c[i][j]);
                check = false;
            }
        }
    }

    if (check) { 
        printf("CUDA CALS CORRECT!!\n");
    }

    

    return 0;
    
}