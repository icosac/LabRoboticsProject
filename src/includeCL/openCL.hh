#ifndef OPENCL_HH
#define OPENCL_HH

#define __CL_ENABLE_EXCEPTIONS

#include "cl.hh"
#ifdef __APPLE__
	#include <OpenCL/opencl.h>
#else
	#include <CL/cl.h>
#endif

#include<iostream>
#include<string>
#include<fstream>

using namespace cl;

void createWorkflow(Context* context,
										std::vector<CommandQueue>* queues,
										std::vector<Device>* devices);

int createProgram (const int type,
										const std::string _source,
										const Context* context,
										std::vector<Device>* devices,
										Program* program);
#endif