#include "openCL.hh"

namespace CL {
    void createWorkflow() {
        int ret = 0;
        std::vector<Platform> platforms;
        Platform::get(&platforms);
        platforms[0].getDevices(CL_DEVICE_TYPE_ALL,
                                &devices); //I suppose there is only 1 platform TODO Check for more platforms

        Context context1(devices);
        context = context1;
        for (Device device : devices) {
            queues.push_back(CommandQueue(context, device));
        }
    }

    int createProgram(const std::string _source,
                      const int type) {

        int ret = 0;
        std::string sourceCode;
        if (type) {
            std::ifstream sourceFile(_source);
            std::string _sourceCode((std::istreambuf_iterator<char>(sourceFile)), std::istreambuf_iterator<char>());
            sourceCode = _sourceCode;
        } else {
            std::string _sourceCode = _source;
            sourceCode = _sourceCode;
        }
        try {
            Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));
            program = Program(context, source);
            // program.build(devices);
            if (program.build(devices) != CL_SUCCESS) {
                std::cerr << "ERRORE" << std::endl;
            }
        }
        catch (Error error) {
            ret = error.err();
            std::cout << "Errore: " << error.what() << "(" << error.err() << ")" << std::endl;
            if (error.err() == CL_BUILD_PROGRAM_FAILURE) {
                for (Device dev : devices) {
                    // Check the build status
                    cl_build_status status = program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(dev);
                    if (status != CL_BUILD_ERROR)
                        continue;
                    // Get the build log
                    std::string name = dev.getInfo<CL_DEVICE_NAME>();
                    std::string buildlog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
                    std::cerr << "Build log for " << name << ":" << std::endl << "\\" << buildlog << "\\" << std::endl;
                }
            } else {
                throw error;
            }
        }
        return ret;
    }

    std::string deviceName (const int ID){
        return devices[ID].getInfo<CL_DEVICE_NAME>();
    }

    std::vector<std::string> deviceInfo (){
        std::vector<std::string> ret;
        for (int i=0; i<devices.size(); i++){
            ret.push_back(deviceInfo(i));
        }
        return ret;
    }

    std::string deviceInfo (const int ID){
        std::stringstream ret;
        if (ID>=0 && ID<devices.size()){
            ret<<"Device " << ID << ": " << std::endl;
            ret<<"\tDevice Name: " << devices[ID].getInfo<CL_DEVICE_NAME>() << std::endl;  
            ret<<"\tDevice Type: " << devices[ID].getInfo<CL_DEVICE_TYPE>();
            ret<<" (GPU: " << CL_DEVICE_TYPE_GPU << ", CPU: " << CL_DEVICE_TYPE_CPU << ")" << std::endl;  
            ret<<"\tDevice Vendor: " << devices[ID].getInfo<CL_DEVICE_VENDOR>() << std::endl;
            ret<<"\tDevice Max Compute Units: " << devices[ID].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
            ret<<"\tDevice Global Memory: " << devices[ID].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << std::endl;
            ret<<"\tDevice Max Clock Frequency: " << devices[ID].getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl;
            ret<<"\tDevice Max Allocateable Memory: " << devices[ID].getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>() << std::endl;
            ret<<"\tDevice Local Memory: " << devices[ID].getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;
            ret<<"\tDevice Available: " << devices[ID].getInfo< CL_DEVICE_AVAILABLE>() << std::endl;
        }
        else {
            ret << "ERROR Wrong DEVICE ID";
        }
        return ret.str();
    }
 }
