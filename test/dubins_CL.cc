#if 1
#include <iostream>
#include <fstream>
#include "maths.hh"
#include "openCL.hh"
#include <chrono>

using namespace std;
using namespace cl;

typedef std::chrono::high_resolution_clock Clock;

Context CL::context;
vector<CommandQueue> CL::queues;
vector<Device> CL::devices;
Program CL::program;

int main (){
  double sum=0.0, avrg=0.0, min=100.0, max=0.0;

  #define DEVICE_ID 2
  FILE* LSL_f=fopen("data/test/CL/2/LSL_CL.test", "w");
  FILE* RSR_f=fopen("data/test/CL/2/RSR_CL.test", "w");
  FILE* LSR_f=fopen("data/test/CL/2/LSR_CL.test", "w");
  FILE* RSL_f=fopen("data/test/CL/2/RSL_CL.test", "w");
  FILE* LRL_f=fopen("data/test/CL/2/LRL_CL.test", "w");
  FILE* RLR_f=fopen("data/test/CL/2/RLR_CL.test", "w");
  FILE* time_f=fopen("data/test/CL/2/time.test" , "a");

	CL::createWorkflow();
	cout << "Contesto creato" << endl;
  cout << CL::deviceName(DEVICE_ID) << endl;
  cout << CL::deviceInfo(DEVICE_ID) << endl;

	Buffer dev_th0=Buffer(CL::context, CL_MEM_READ_ONLY, sizeof(double));
	Buffer dev_th1=Buffer(CL::context, CL_MEM_READ_ONLY, sizeof(double));
	Buffer dev_kmax=Buffer(CL::context, CL_MEM_READ_ONLY, sizeof(double));
  Buffer dev_LSL=Buffer(CL::context, CL_MEM_READ_WRITE, sizeof(double)*3);  
  Buffer dev_LSR=Buffer(CL::context, CL_MEM_READ_WRITE, sizeof(double)*3);  
  Buffer dev_RSL=Buffer(CL::context, CL_MEM_READ_WRITE, sizeof(double)*3);  
  Buffer dev_RSR=Buffer(CL::context, CL_MEM_READ_WRITE, sizeof(double)*3);  
  Buffer dev_RLR=Buffer(CL::context, CL_MEM_READ_WRITE, sizeof(double)*3);  
	Buffer dev_LRL=Buffer(CL::context, CL_MEM_READ_WRITE, sizeof(double)*3);	

	CL::createProgram("src/dubins.cl");
	cout << "CL::Programma creato" << endl;
	
	fprintf(LSL_f, "th0, th1, kmax, sc_s1, sc_s2, sc_s3\n");
  fprintf(RSR_f, "th0, th1, kmax, sc_s1, sc_s2, sc_s3\n");
  fprintf(LSR_f, "th0, th1, kmax, sc_s1, sc_s2, sc_s3\n");
  fprintf(RSL_f, "th0, th1, kmax, sc_s1, sc_s2, sc_s3\n");
  fprintf(LRL_f, "th0, th1, kmax, sc_s1, sc_s2, sc_s3\n");
  fprintf(RLR_f, "th0, th1, kmax, sc_s1, sc_s2, sc_s3\n");

  int i=0;
  for (double th0=0.0; th0<=2*M_PI; th0+=0.05){
   	CL::queues[DEVICE_ID].enqueueWriteBuffer(dev_th0, CL_TRUE, 0, sizeof(double), &th0);
    for (double th1=0.0; th1<=2*M_PI; th1+=0.05){
			CL::queues[DEVICE_ID].enqueueWriteBuffer(dev_th1, CL_TRUE, 0, sizeof(double), &th1);
    	for (double kmax=0.0; kmax<=5; kmax+=0.05){
		    CL::queues[DEVICE_ID].enqueueWriteBuffer(dev_kmax, CL_TRUE, 0, sizeof(double), &kmax);
		    
        // printf("%f, %f, %f\n", th0, th1, kmax);
        double **ret=new double* [6];
        for (int a=0; a<6; a++){
          ret[a]=new double[3];
        }

        auto t1=Clock::now();
        Kernel LSL(CL::program, "LSL");
				LSL.setArg(0, dev_th0);
				LSL.setArg(1, dev_th1);
				LSL.setArg(2, dev_kmax);
				LSL.setArg(3, dev_LSL);

        Kernel RSR(CL::program, "RSR");
        RSR.setArg(0, dev_th0);
        RSR.setArg(1, dev_th1);
        RSR.setArg(2, dev_kmax);
        RSR.setArg(3, dev_RSR);

        Kernel LSR(CL::program, "LSR");
        LSR.setArg(0, dev_th0);
        LSR.setArg(1, dev_th1);
        LSR.setArg(2, dev_kmax);
        LSR.setArg(3, dev_LSR);

        Kernel RSL(CL::program, "RSL");
        RSL.setArg(0, dev_th0);
        RSL.setArg(1, dev_th1);
        RSL.setArg(2, dev_kmax);
        RSL.setArg(3, dev_RSL);

        Kernel LRL(CL::program, "LRL");
        LRL.setArg(0, dev_th0);
        LRL.setArg(1, dev_th1);
        LRL.setArg(2, dev_kmax);
        LRL.setArg(3, dev_LRL);

        Kernel RLR(CL::program, "RLR");
        RLR.setArg(0, dev_th0);
        RLR.setArg(1, dev_th1);
        RLR.setArg(2, dev_kmax);
        RLR.setArg(3, dev_RLR);

				NDRange global(1);
				NDRange local(1);

        try{
          CL::queues[DEVICE_ID].enqueueNDRangeKernel(LSL, NullRange, global, local);
          CL::queues[DEVICE_ID].enqueueNDRangeKernel(RSR, NullRange, global, local);
          CL::queues[DEVICE_ID].enqueueNDRangeKernel(LSR, NullRange, global, local);
          CL::queues[DEVICE_ID].enqueueNDRangeKernel(RSL, NullRange, global, local);
          CL::queues[DEVICE_ID].enqueueNDRangeKernel(LRL, NullRange, global, local);
  				CL::queues[DEVICE_ID].enqueueNDRangeKernel(RLR, NullRange, global, local);
        } catch (Error error) {
          std::cout << "Error: " << error.what() << "(" << error.err() << ")" << std::endl;
          throw error;
        }
        auto t2=Clock::now();

        CL::queues[DEVICE_ID].enqueueReadBuffer(dev_LSL, CL_TRUE, 0, sizeof(double)*3, ret[0]);
        CL::queues[DEVICE_ID].enqueueReadBuffer(dev_RSR, CL_TRUE, 0, sizeof(double)*3, ret[1]);
        CL::queues[DEVICE_ID].enqueueReadBuffer(dev_LSR, CL_TRUE, 0, sizeof(double)*3, ret[2]);
        CL::queues[DEVICE_ID].enqueueReadBuffer(dev_RSL, CL_TRUE, 0, sizeof(double)*3, ret[3]);
        CL::queues[DEVICE_ID].enqueueReadBuffer(dev_LRL, CL_TRUE, 0, sizeof(double)*3, ret[4]);
				CL::queues[DEVICE_ID].enqueueReadBuffer(dev_RLR, CL_TRUE, 0, sizeof(double)*3, ret[5]);
        
        double diff=(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count())/1000000.0;
        i++;
        sum+=diff;
        max=(max>diff ? max : diff);
        min=(min<diff ? min : diff);

        fprintf(LSL_f, "%f, %f, %f, %f, %f, %f\n", th0, th1, kmax, ret[0][0], ret[0][1], ret[0][2]);
        fprintf(RSR_f, "%f, %f, %f, %f, %f, %f\n", th0, th1, kmax, ret[1][0], ret[1][1], ret[1][2]);
        fprintf(LSR_f, "%f, %f, %f, %f, %f, %f\n", th0, th1, kmax, ret[2][0], ret[2][1], ret[2][2]);
        fprintf(RSL_f, "%f, %f, %f, %f, %f, %f\n", th0, th1, kmax, ret[3][0], ret[3][1], ret[3][2]);
        fprintf(LRL_f, "%f, %f, %f, %f, %f, %f\n", th0, th1, kmax, ret[4][0], ret[4][1], ret[4][2]);
        fprintf(RLR_f, "%f, %f, %f, %f, %f, %f\n", th0, th1, kmax, ret[5][0], ret[5][1], ret[5][2]);

				// if (i==20){
				// 	return 0;
				// }
			}
		}
	}
  
  avrg=sum/i;
  fprintf(time_f, "LSL+RSR+LSR+RSL+LRL+RLR\nTot: %fms, avrg: %fms, min: %fms, max: %fms\n\n", sum, avrg, min, max);

	return 0;
}

#else 


#define __CL_ENABLE_EXCEPTIONS

#include "cl.hh"
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

		vector<Device> CL::devices;
		platforms[0].getCL::Devices(CL_DEVICE_TYPE_ALL, &CL::devices);
    for (auto device : CL::devices){
      int *A=new int[ELEMENTS];
      int *B=new int[ELEMENTS];
      int *C=new int[ELEMENTS];
      for (int i=0; i<ELEMENTS; i++){
        A[i]=i;
        B[i]=ELEMENTS-i;
      }

      string value;
      device.getInfo(CL_DEVICE_NAME, &value);
      ofstream fileout; //fileout.open(("test/"+value+".txt").c_str(), ios::app);
      
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

      const auto beginCL::Context = high_resolution_clock::now(); 
      CL::Context CL::context(CL::devices);
  		CommandQueue queue = CommandQueue(CL::context, device);
      
  		Buffer bufferA=Buffer(CL::context, CL_MEM_READ_ONLY, datasize);
  		Buffer bufferB=Buffer(CL::context, CL_MEM_READ_ONLY, datasize);
  		Buffer bufferC=Buffer(CL::context, CL_MEM_READ_ONLY, datasize);
  		
  		queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, datasize, A);
  		queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, datasize, B);
      const auto endCL::Context = high_resolution_clock::now(); 
      auto CL::contextTime=duration<double, std::milli>(endCL::Context-beginCL::Context).count();

  		ifstream sourceFile("vectoradd.cl");
  		string sourceCode(istreambuf_iterator<char>(sourceFile), (istreambuf_iterator<char>()));
  		
      const auto beginCL::ProgramC = high_resolution_clock::now();
      CL::Program::Sources source (1, make_pair (sourceCode.c_str(), sourceCode.length()+1));
  		CL::Program CL::program = CL::Program(CL::context, source);
      CL::program.build(CL::devices);
      const auto endCL::ProgramC = high_resolution_clock::now();
      auto CL::programCTime=duration<double, std::milli>(endCL::ProgramC-beginCL::ProgramC).count();

      const auto beginKernelC = high_resolution_clock::now();
  		Kernel vector_add(CL::program, "vector_add");
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

      auto total=CL::contextTime+CL::programCTime+kernelCTime+kernelTime;

  		queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, datasize, C);
  		for (int i=0; i<ELEMENTS; i++){
  			if (C[i]!=ELEMENTS){
  				goto ERROR; 
  			}
  		}
  		fileout << ELEMENTS << ": " << total << "ms,\t" << CL::contextTime << "ms,\t" << CL::programCTime << "ms,\t" << kernelCTime << "ms,\t" << kernelTime << "ms" << endl;
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


#endif