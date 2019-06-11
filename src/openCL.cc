#include "openCL.hh"

void createWorkflow (	Context* context,
											std::vector<CommandQueue>* queues,
											std::vector<Device>* devices)
{
	std::vector<Platform> platforms;
	Platform::get(&platforms);
	platforms[0].getDevices(CL_DEVICE_TYPE_ALL, devices); //I suppose there is only 1 platform TODO Check for more platforms
	
	Context context1(*devices);
	*context=context1;
	for (Device device : *devices){
		queues->push_back(CommandQueue(*context, device));
	}
}
 
int createProgram (const int type,
										const std::string _source,
										const Context* context,
										std::vector<Device>* devices,
										Program* program)
{
	int ret=0;
	std::string sourceCode;
	if (type) {
		std::ifstream sourceFile(_source);
		std::string _sourceCode((std::istreambuf_iterator<char>(sourceFile)), std::istreambuf_iterator<char>());
		sourceCode=_sourceCode;
	}
	else {
		std::string _sourceCode=_source;
		sourceCode=_sourceCode;
	}
	try {
		Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length()+1));
		*program = Program (*context, source);
		// program->build(*devices);
		if (program->build(*devices)!=CL_SUCCESS){
			std::cerr << "ERRORE" << std::endl;
		}
	} 
	catch (Error error){
		ret=error.err();
		std::cout << "Errore: " << error.what() << "(" << error.err() << ")" << std::endl;
		if (error.err() == CL_BUILD_PROGRAM_FAILURE)
	  {
	    for (Device dev : *devices)
	    {
	      // Check the build status
	      cl_build_status status = program->getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
	      if (status != CL_BUILD_ERROR)
	        continue;
	      // Get the build log
	      std::string name     = dev.getInfo<CL_DEVICE_NAME>();
	      std::string buildlog = program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
	      std::cerr << "Build log for " << name << ":" << std::endl << "\\" << buildlog << "\\" << std::endl;
	    }
	  }
	  else
	  {
	    throw error;
	  }
	}
	return ret;
}

