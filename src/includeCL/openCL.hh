#ifndef OPENCL_HH
#define OPENCL_HH

#define __CL_ENABLE_EXCEPTIONS
#define CL_SILENCE_DEPRECATION

#include "cl.hh"
#ifdef __APPLE__
	#include <OpenCL/opencl.h>
#else
	#include <CL/cl.h>
#endif

#include<iostream>
#include<string>
#include<fstream>
#include<sstream>

using namespace cl;

namespace CL
{
    extern cl::Context context;
    extern std::vector<CommandQueue> queues;
    extern std::vector<Device> devices;
    extern cl::Program program;

    void createWorkflow();

    int createProgram(const std::string _source,
                      const int type=1);

    std::string deviceName (const int ID);

    std::vector<std::string> deviceInfo ();

    std::string deviceInfo (const int ID);
}


#endif