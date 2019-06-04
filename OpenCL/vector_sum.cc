#define CL_SILENCE_DEPRECATION

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <typeinfo>
#include <chrono>
#include <cmath>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
 
#define MAX_SOURCE_SIZE (0x100000)
#define DEVICE_ID 0

using namespace std;
using namespace std::chrono;

int main(void) {
  // Create the two input vectors
  int i;
  const int LIST_SIZE = 1048576;
  cout << pow(2, sizeof(int)) << endl;
  int *A = (int*)malloc(sizeof(int)*LIST_SIZE);
  int *B = (int*)malloc(sizeof(int)*LIST_SIZE);
  for(i = 0; i < LIST_SIZE; i++) {
    A[i] = i;
    B[i] = LIST_SIZE - i;
  }
  int *D = (int*)malloc(sizeof(int)*LIST_SIZE);
  const auto begin1 = high_resolution_clock::now(); 
  for (i=0; i<LIST_SIZE; i++){
    D[i]=A[i]+B[i];
  }
  auto time1 = high_resolution_clock::now() - begin1;
  std::cout << "Elapsed CPU time: " << duration<double, std::milli>(time1).count() << "ms.\n";

  // Load the kernel source code into the array source_str
  FILE *fp;
  char *source_str;
  size_t source_size;

  fp = fopen("vectoradd.cl", "r");
  if (!fp) {
      fprintf(stderr, "Failed to load kernel.\n");
      exit(1);
  }
  source_str = (char*)malloc(MAX_SOURCE_SIZE);
  source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose( fp );

  // Get platform and device information
  cl_platform_id platform_id = NULL;
  cl_device_id* device_id = NULL;
  cl_uint deviceCount=0;
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount);
  device_id=(cl_device_id*) malloc (sizeof(cl_device_id)*deviceCount);
  ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_GPU, deviceCount, device_id, &ret_num_devices);
  size_t valueSize;
  clGetDeviceInfo(device_id[DEVICE_ID], CL_DEVICE_NAME, 0, NULL, &valueSize);
  char* value=(char*) malloc (valueSize);
  clGetDeviceInfo(device_id[DEVICE_ID], CL_DEVICE_NAME, valueSize, value, NULL);
  printf("Nome: %s\n", value);
  free(value);
  free(device_id);
  // Create an OpenCL context
  cl_device_id device = device_id[DEVICE_ID];
  cl_context context = clCreateContext( NULL, 1, &device, NULL, NULL, &ret);

  // Create a command queue
  cl_command_queue command_queue = clCreateCommandQueue(context, device, 0, &ret);

  // Create memory buffers on the device for each vector 
  cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, LIST_SIZE * sizeof(int), NULL, &ret);
  cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, LIST_SIZE * sizeof(int), NULL, &ret);
  cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, LIST_SIZE * sizeof(int), NULL, &ret);

  // Copy the lists A and B to their respective memory buffers
  ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0, LIST_SIZE * sizeof(int), A, 0, NULL, NULL);
  ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, LIST_SIZE * sizeof(int), B, 0, NULL, NULL);

  // Create a program from the kernel source
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);

  // Build the program
  ret = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

  // Create the OpenCL kernel
  cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);

  // Set the arguments of the kernel
  ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
  ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
  ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);

  // Execute the OpenCL kernel on the list
  size_t global_item_size = LIST_SIZE; // Process the entire lists
  size_t local_item_size = 32; // Divide work items into groups of 64
  const auto begin = high_resolution_clock::now(); 
  ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
  auto time = high_resolution_clock::now() - begin;
  std::cout << "Elapsed time: " << duration<double, std::milli>(time).count() << "ms.\n";
  // Read the memory buffer C on the device to the local variable C
  int *C = (int*)malloc(sizeof(int)*LIST_SIZE);
  ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0, LIST_SIZE * sizeof(int), C, 0, NULL, NULL);

  // Display the result to the screen
  // for(i = 0; i < LIST_SIZE; i++)
  //     printf("%d + %d = %d\n", A[i], B[i], C[i]);

  // Clean up
  ret = clFlush(command_queue);
  ret = clFinish(command_queue);
  ret = clReleaseKernel(kernel);
  ret = clReleaseProgram(program);
  ret = clReleaseMemObject(a_mem_obj);
  ret = clReleaseMemObject(b_mem_obj);
  ret = clReleaseMemObject(c_mem_obj);
  ret = clReleaseCommandQueue(command_queue);
  ret = clReleaseContext(context);

  free(A);
  free(B);
  free(C);
  return 0;
}