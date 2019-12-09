/*
This program will numerically compute the integral of
                  4/(1+x*x)
from 0 to 1.  The value of this integral is pi -- which
is great since it gives us an easy way to check the answer.
The is the original sequential program.  It uses the timer
from the OpenMP runtime library
History: Written by Tim Mattson, 11/99.
*/
#include <stdio.h>
#include "cuda.h"
/*#include <omp.h>*/
static long num_steps = 1000000000;
double step;
double pi;


__global__ void calculation (int num_steps,int step,double*tabsum){

	int i;
	double x, sum= 0.0;
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int num_thread = gridDim.x*blockDim.x;

	for (i=tid;i<= num_steps; i+=num_thread){
		x = (i-0.5)*step;
		sum = sum + 4.0/(1.0+x*x);
	}

	__syncthreads();
	*(tabsum + blockIdx.x - 1) = sum;
}


int main ()
{
	  cudaEvent_t start_time, stop_time;
		float elapsed_time;
		double *tabsum=0;

		step = 1.0/(double) num_steps;

		cudaEventCreate( &start_time );
    cudaEventCreate( &stop_time );
    cudaEventRecord( start_time, 0 );
/*	  start_time = omp_get_wtime();*/
	  calculation<<<16,16>>>(num_steps,step,*tabsum);

		for (int i=0;i<blockDim.x;i++){

			pi = **(tabsum + i) * step;
		}

		cudaEventRecord(stop_time,0);
    cudaEventSynchronize( stop_time );
		cudaEventElapsedTime(&elapsed_time,start_time,stop_time);
/*	  run_time = omp_get_wtime() - start_time;*/
	  printf("\n pi with %ld steps is %lf in %lf millisecond\n ",num_steps,pi,elapsed_time);
		cudaEventDestroy( start_time );
    cudaEventDestroy( stop_time );
}
