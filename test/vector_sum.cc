#define __CL_ENABLE_EXCEPTIONS

#include "cl.hpp"
#include "openCL.hh"

#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<chrono>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// #define ELEMENTS 2048

using namespace std;
using namespace cl;
using namespace std::chrono;

int main (int argc, char* argv[]){
	const unsigned long int ELEMENTS=atoi(argv[1]);
  cout << ELEMENTS << endl;
  unsigned long int datasize=sizeof(int)*ELEMENTS;
	try {
		vector<Platform> platforms;
		Platform::get(&platforms);

		vector<Device> devices;
		platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
    for (auto device : devices){
      int *A=new int[ELEMENTS];
      int *B=new int[ELEMENTS];
      int *C=new int[ELEMENTS];
      for (int i=0; i<ELEMENTS; i++){
        A[i]=i;
        B[i]=ELEMENTS-i;
      }

      string value;
      device.getInfo(CL_DEVICE_NAME, &value);
      ofstream fileout; fileout.open(("test/"+value+".txt").c_str(), ios::app);
      
      // cout << "NAME: " << value << endl;
      cl_uint max_work_item_dimensions;
      device.getInfo(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, &max_work_item_dimensions);
      // cout << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: " << max_work_item_dimensions << endl;
      std::size_t max_work_item_sizes[3]={1,2,3};
      device.getInfo(CL_DEVICE_MAX_WORK_ITEM_SIZES, &max_work_item_sizes);
      // cout << "CL_DEVICE_MAX_WORK_ITEM_SIZES: "; 
      // for (cl_uint i = 0; i < max_work_item_dimensions; ++i) {
      //   cout << max_work_item_sizes[i] << endl; 
      // }

      const auto beginContext = high_resolution_clock::now(); 
      Context context(devices);
  		CommandQueue queue = CommandQueue(context, device);
      
  		Buffer bufferA=Buffer(context, CL_MEM_READ_ONLY, datasize);
  		Buffer bufferB=Buffer(context, CL_MEM_READ_ONLY, datasize);
  		Buffer bufferC=Buffer(context, CL_MEM_READ_ONLY, datasize);
  		
  		queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, datasize, A);
  		queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, datasize, B);
      const auto endContext = high_resolution_clock::now(); 
      auto contextTime=duration<double, std::milli>(endContext-beginContext).count();

  		ifstream sourceFile("vectoradd.cl");
  		string sourceCode(istreambuf_iterator<char>(sourceFile), (istreambuf_iterator<char>()));
  		
      const auto beginProgramC = high_resolution_clock::now();
      Program::Sources source (1, make_pair (sourceCode.c_str(), sourceCode.length()+1));
  		Program program = Program(context, source);
      program.build(devices);
      const auto endProgramC = high_resolution_clock::now();
      auto programCTime=duration<double, std::milli>(endProgramC-beginProgramC).count();

      const auto beginKernelC = high_resolution_clock::now();
  		Kernel vector_add(program, "vector_add");
  		vector_add.setArg(0, bufferA);
  		vector_add.setArg(1, bufferB);
  		vector_add.setArg(2, bufferC);

  		NDRange global(4);
  		NDRange local(2);
  		const auto endKernelC = high_resolution_clock::now();
      auto kernelCTime=duration<double, std::milli>(endKernelC-beginKernelC).count();
      
      const auto beginKernel = high_resolution_clock::now();
      queue.enqueueNDRangeKernel(vector_add, NullRange, global, local);
      const auto endKernel = high_resolution_clock::now();
      auto kernelTime=duration<double, std::milli>(endKernel-beginKernel).count();

      auto total=contextTime+programCTime+kernelCTime+kernelTime;

  		queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, datasize, C);
  		for (int i=0; i<ELEMENTS; i++){
  			if (C[i]!=ELEMENTS){
  				goto ERROR; 
  			}
  		}
  		fileout << ELEMENTS << ": " << total << "ms,\t" << contextTime << "ms,\t" << programCTime << "ms,\t" << kernelCTime << "ms,\t" << kernelTime << "ms" << endl;
  		goto END;
      ERROR:
      fileout << ELEMENTS << " ERRORE" << endl;
      END: 
  		delete(A);
  		delete(B);
  		delete(C);

      fileout.close();
    }
	}
	catch (Error error){
		cout << "Errore: " << error.what() << "(" << error.err() << ")" << endl;
	}

  return 0;

}