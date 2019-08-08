// #include<iostream>
// #include<string.h>
// #include<cmath>

// using namespace std;

// #define THREAD 256
// #define WIDTH 5
// #define HEIGHT 2
// #define SIZE WIDTH*HEIGHT

// __global__ void matrixAdd (int* A, int* B, int* C, size_t pitchA, size_t pitchB, size_t pitchC, int w, int h){
// 	int x=threadIdx.x;
// 	int y=threadIdx.y;
// 	// int* rowA = (int*)((char*)A + x * pitchA);
// 	// int* rowB = (int*)((char*)B + x * pitchB);
// 	// int* rowC = (int*)((char*)C + x * pitchC);
// 	// rowC[y]=rowA[y]+rowB[y];
// }

// __global__ void printMat(int* A, size_t pitchA){
// 	int x=threadIdx.x;
// 	int y=threadIdx.y;
// 	if (x<HEIGHT && y<WIDTH){
// 		printf("A[%d, %d]=%d, ", x, y, A[x*pitchA/sizeof(int)+y]);
// 	}
// }

// int main(){
// 	{cudaFree(0);
// 		int* dev=(int*) malloc(sizeof(int));
// 		cudaGetDevice(dev);
	
// 		cudaDeviceProp prop;
// 		cudaGetDeviceProperties(&prop, *dev);
	
// 		printf("name: %s\n", prop.name);
// 		printf("totalGlobalMem: %d\n", prop.totalGlobalMem);
// 		printf("sharedMemPerBlock: %d\n", prop.sharedMemPerBlock);
// 		printf("regsPerBlock: %d\n", prop.regsPerBlock);
// 		printf("warpSize: %d\n", prop.warpSize);
// 		printf("memPitch: %d\n", prop.memPitch);
// 		printf("maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
// 		printf("maxThreadsDim %d, %d, %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
// 		printf("maxGridSize %d, %d, %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
// 		printf("totalConstMem: %d\n", prop.totalConstMem);
// 		printf("major: %d\n", prop.major);
// 		printf("minor: %d\n", prop.minor);
// 		printf("clockRate: %d\n", prop.clockRate);
// 		printf("textureAlignment: %d\n", prop.textureAlignment);
// 		printf("deviceOverlap: %d\n", prop.deviceOverlap);
// 		printf("multiProcessorCount: %d\n", prop.multiProcessorCount);
// 		printf("kernelExecTimeoutEnabled: %d\n", prop.kernelExecTimeoutEnabled);
// 		printf("integrated: %d\n", prop.integrated);
// 		printf("canMapHostMemory: %d\n", prop.canMapHostMemory);
// 		printf("computeMode: %d\n", prop.computeMode);
// 		printf("concurrentKernels: %d\n", prop.concurrentKernels);
// 		printf("ECCEnabled: %d\n", prop.ECCEnabled);
// 		printf("pciBusID: %d\n", prop.pciBusID);
// 		printf("pciDeviceID: %d\n", prop.pciDeviceID);
// 		printf("tccDriver: %d\n", prop.tccDriver);
// 	}

// 	int* A=(int*) malloc(sizeof(int)*SIZE);
// 	int* B=(int*) malloc(sizeof(int)*SIZE);
// 	int* C=(int*) malloc(sizeof(int)*SIZE);

// 	for (int i=0; i<SIZE; i++){
// 		A[i]=i+1;
// 		B[i]=SIZE-i-1;
// 	}

// 	for (int i=0; i<SIZE; i++){
// 		cout << A[i] << ", ";
// 		if ((i+1)%WIDTH==0){ cout << endl; }
// 	}

// 	for (int i=0; i<SIZE; i++){
// 		cout << B[i] << ", ";
// 		if ((i+1)%WIDTH==0){ cout << endl; }
// 	}

// 	size_t pitchA, pitchB, pitchC;

// 	int* dev_A; cudaMallocPitch(&dev_A, &pitchA, WIDTH*sizeof(int), HEIGHT);
// 	cout << "pitch: " << pitchA/sizeof(int) << endl;
// 	// double* dev_B; cudaMallocPitch(&dev_B, &pitchB, WIDTH*sizeof(double), HEIGHT);
// 	// double* dev_C; cudaMallocPitch(&dev_C, &pitchC, WIDTH*sizeof(double), HEIGHT);

// 	cudaMemcpy2D(dev_A, pitchA, A, WIDTH*sizeof(int), WIDTH*sizeof(int), HEIGHT, cudaMemcpyHostToDevice);

// 	dim3 dim(HEIGHT, WIDTH);
// 	printMat<<<1, dim>>> (dev_A, pitchA);

// 	cudaFree(dev_A);

// 	// // func <<<(int)(SIZE/THREAD)+1, (THREAD)>>> (a);
// 	// int err=cudaDeviceSynchronize();
// 	// printf("%d\n", err);

// 	// // cudaMemcpy(t_h, t[SIZE-1], DIM*sizeof(int), cudaMemcpyDeviceToHost);

// 	// cudaFree(a);

// 	// free(t_h);
// 	return 0;
// }