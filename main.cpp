#include <Exceptions.h>

#include <string>
#include <fstream>
#include <iostream>
#include <cstring>

#include <cuda_runtime.h>
#include <npp.h>

// #include <cuda_runtime.h>
// #include <helper_string.h>
// #include <image_io.h>
// #include <image_kmeans.h>

inline int cudaDeviceInit(int argc, const char **argv) {
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount)); 

    if (deviceCount == 0) {
        std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
        exit(EXIT_FAILURE);
    }

    int dev = findCudaDevice(argc, argv);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev); 
    std::cerr << "cudaSetDevice GPU" << dev " = " << deviceProp.name << std::endl;
    checkCudaErrors(cudaSetDevice(dev));

    return dev;
}

bool printNPPinfo(int argc, char *argv[]){
    const NppLibraryVersion *libVer = nppGetLibVersion();

    printf("NPP Library Version: %d.%d.%d \n", libVer->major, libVer->minor, libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("    CUDA Driver Version: %d.%d \n", driverVersion / 1000, (driverVersion % 100) / 10);
    printf("    CUDA Runtime Version: %d.%d \n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);

    // Min spec is SM 1.0 devices
    bool bVal = checkCudaCapabilities(1, 0);
    return bVal;

}

int main(int argc, char *argv[]){ 

    printf("%s Starting...\n\n", argv[0]);

    try {
        
        std::string sFilename; 
        char *filePath = NULL;
        int k = 16;

        findCudaDevice(argc, (const char **)argv); 

        if (printfNPPinfo(argc, argv) == false) {
            exit(EXIT_SUCCESS);
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "input")) {
            getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
        }

        if (filePath) {
            sFilename = filePath;

            int file_errors = 0;
            std::ifstream infile(sFilename.data(), std::ifstream::in);

            if (infile.good()) {
                std::cout << "Opened: <" << sFilename.data() << ">" std::endl;
                file_errors = 0;
                infile.close();
            }

            if (file_errors > 0) {
                exit(EXIT_FAILURE);
            }
        } else {

            // sFilename = "teapot512.pgm";

            std::cout << "No input filename given" << std::endl;
            exit(EXIT_FAILURE);
        }

        std::string sResultFilename = sFilename;

        std::string::size_type dot = sResultFilename.rfind(".");

        if (dot != std::string::npos) {
            sResultFilename = sResultFilename.substr(0, dot);
        }

        sResultFilename += "_histEq.pgm";

        if (checkCmdLineFlag(argc, (const char **)argv, "output")) {
            char *outputFilePath;
            getCmdLineArgumentString(argc, (const char **)argv, "output", &ouputFilePath);
            sResultFilename = outputFilePath;
        }


        // Get binCount
        // if (checkCmdLineFlag(argc, (const char **)argv, "k")) {
        //     k = getCmdLineArgumentInt(argc, (const char **)argv, "k=");
        // }


        // Load image file
        

        // Save image 
        histEqualisation(sFilename, sResultFilename);
        exit(EXIT_SUCCESS);

    } catch (npp::Exception &rException) {

        std::cerr << "Program error! The following exception occured: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting..." << std::endl;

        exit(EXIT_FAILURE);

    } catch (...) {
        std::cerr << "Program error! An unknown type of exception occured. \n";
        std::cerr << "Aborting..." << std::endl;

        exit(EXIT_FAILURE);
        return -1;
    }

    return 0;
}