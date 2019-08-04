#include<iostream>
#include<string.h>
#include<cmath>

using namespace std;

#define SIZE 10000000000
#define THREAD 256
#define BASE 18
#define DIM (int)(log(SIZE)/log(BASE)+1)

// __device__ int** t;

__device__ void toBase(int val, int* ret){
	int i=0;
	while(val>0){
		ret[i]=val%BASE;
		val=(int)(val/BASE);
		i++;
	}
}

__global__ void func(int* a){
	int pidx=blockIdx.x*blockDim.x+threadIdx.x;
	if (pidx<SIZE){
		toBase(pidx, a);
		// t[pidx]=a;
		if (pidx>SIZE-10 && pidx<SIZE){
			printf("%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n", pidx, 
			a[0], a[1], a[2], a[3], a[4], a[5], a[6],
			 a[7], a[8], a[9]);
		}
	}
}

int main(){
	cudaFree(0);
	int* dev=(int*) malloc(sizeof(int));
	cudaGetDevice(dev);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, *dev);

	printf("name: %s\n", prop.name);
	printf("totalGlobalMem: %d\n", prop.totalGlobalMem);
	printf("sharedMemPerBlock: %d\n", prop.sharedMemPerBlock);
	printf("regsPerBlock: %d\n", prop.regsPerBlock);
	printf("warpSize: %d\n", prop.warpSize);
	printf("memPitch: %d\n", prop.memPitch);
	printf("maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
	printf("maxThreadsDim %d, %d, %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("maxGridSize %d, %d, %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf("totalConstMem: %d\n", prop.totalConstMem);
	printf("major: %d\n", prop.major);
	printf("minor: %d\n", prop.minor);
	printf("clockRate: %d\n", prop.clockRate);
	printf("textureAlignment: %d\n", prop.textureAlignment);
	printf("deviceOverlap: %d\n", prop.deviceOverlap);
	printf("multiProcessorCount: %d\n", prop.multiProcessorCount);
	printf("kernelExecTimeoutEnabled: %d\n", prop.kernelExecTimeoutEnabled);
	printf("integrated: %d\n", prop.integrated);
	printf("canMapHostMemory: %d\n", prop.canMapHostMemory);
	printf("computeMode: %d\n", prop.computeMode);
	printf("concurrentKernels: %d\n", prop.concurrentKernels);
	printf("ECCEnabled: %d\n", prop.ECCEnabled);
	printf("pciBusID: %d\n", prop.pciBusID);
	printf("pciDeviceID: %d\n", prop.pciDeviceID);
	printf("tccDriver: %d\n", prop.tccDriver);

	int* a; cudaMalloc((void**)&a, sizeof(int)*DIM);
	// int** t; cudaMalloc((void**)t, )
	// int* t_h; t_h=(int*) malloc(sizeof(int)*DIM);

	func <<<(int)(SIZE/THREAD)+1, (THREAD)>>> (a);
	int err=cudaDeviceSynchronize();
	printf("%d\n", err);

	// cudaMemcpy(t_h, t[SIZE-1], DIM*sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(a);

	// free(t_h);
	return 0;
}