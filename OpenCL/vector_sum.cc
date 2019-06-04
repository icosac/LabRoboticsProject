#define __CL_ENABLE_EXCEPTIONS

#include<CL/cl.hpp>
#include<iostream>
#include<fstream>
#include<string>
#include<vector>

#define ELEMENTS 2048

using namespace std;
using namespace cl;

int main (){
	unsigned long int datasize=sizeof(int)*ELEMENTS;
	int *A=new int[ELEMENTS];
	int *B=new int[ELEMENTS];
	int *C=new int[ELEMENTS];
	for (int i=0; i<ELEMENTS; i++){
		A[i]=i;
		B[i]=ELEMENTS-i;
	}

	try {
		vector<Platform> platforms;
		Platform::get(&platforms);
		cout << "ok" << endl;

		vector<Device> devices;
		platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devices);
		

		Context context(devices);
		CommandQueue queue = CommandQueue(context, devices[0]);

		Buffer bufferA=Buffer(context, CL_MEM_READ_ONLY, datasize);
		Buffer bufferB=Buffer(context, CL_MEM_READ_ONLY, datasize);
		Buffer bufferC=Buffer(context, CL_MEM_READ_ONLY, datasize);
		
		queue.enqueueWriteBuffer(bufferA, CL_TRUE, 0, datasize, A);
		queue.enqueueWriteBuffer(bufferB, CL_TRUE, 0, datasize, B);

		ifstream sourceFile("vectoradd.cl");
		string sourceCode(istreambuf_iterator<char>(sourceFile), (istreambuf_iterator<char>()));
		cout << endl << endl << sourceCode << endl << endl;
		Program::Sources source (1, make_pair (sourceCode.c_str(), sourceCode.length()+1));

		Program program = Program(context, source);

		program.build(devices);

		Kernel vector_add(program, "vector_add");

		vector_add.setArg(0, bufferA);
		vector_add.setArg(1, bufferB);
		vector_add.setArg(2, bufferC);

		NDRange global(ELEMENTS);
		NDRange local(256);
		queue.enqueueNDRangeKernel(vector_add, NullRange, global, local);

		queue.enqueueReadBuffer(bufferC, CL_TRUE, 0, datasize, C);
		for (int i=0; i<ELEMENTS; i++){
			if (C[i]!=ELEMENTS){
				goto ERROR; 
			}
		}
		cout << "OK" << endl;
		ERROR:

		delete(A);
		delete(B);
		delete(C);
	}
	catch (Error error){
		cout << "Errore: " << error.what() << "(" << error.err() << ")" << endl;
	}




}