/*****
 * Histogram Eqalisation using NPP 
 */

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string.h>
#include <fstream>
#include <iostream>

#include <cuda_runtime.h>
#include <npp.h>

#include <helper_cuda.h>
#include <helper_string.h>

bool printfNPPinfo(int argc, char *argv[])
{
  const NppLibraryVersion *libVer = nppGetLibVersion();

  printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
         libVer->build);

  int driverVersion, runtimeVersion;
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);

  printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
         (driverVersion % 100) / 10);
  printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
         (runtimeVersion % 100) / 10);

  // Min spec is SM 1.0 devices
  bool bVal = checkCudaCapabilities(1, 0);
  return bVal;
}

int main(int argc, char *argv[])
{
  printf("%s Starting...\n\n", argv[0]);

  try
  {
    std::string sFilename;
    char *filePath;

    findCudaDevice(argc, (const char **)argv);

    if (printfNPPinfo(argc, argv) == false)
    {
      exit(EXIT_SUCCESS);
    }

    if (checkCmdLineFlag(argc, (const char **)argv, "input"))
    {
      getCmdLineArgumentString(argc, (const char **)argv, "input", &filePath);
    }
    else
    {
      filePath = sdkFindFilePath("landscape.pgm", argv[0]);
    }

    if (filePath)
    {
      sFilename = filePath;
    }
    else
    {
      sFilename = "landscape.pgm";
    }

    // if we specify the filename at the command line, then we only test
    // sFilename[0].
    int file_errors = 0;
    std::ifstream infile(sFilename.data(), std::ifstream::in);

    if (infile.good())
    {
      std::cout << "histEqualisation opened: <" << sFilename.data()
                << "> successfully!" << std::endl;
      file_errors = 0;
      infile.close();
    }
    else
    {
      std::cout << "histEqualisation unable to open: <" << sFilename.data() << ">"
                << std::endl;
      file_errors++;
      infile.close();
    }

    if (file_errors > 0)
    {
      exit(EXIT_FAILURE);
    }

    std::string sResultFilename = sFilename;

    std::string::size_type dot = sResultFilename.rfind('.');

    if (dot != std::string::npos)
    {
      sResultFilename = sResultFilename.substr(0, dot);
    }

    sResultFilename += "_histEq.pgm";

    if (checkCmdLineFlag(argc, (const char **)argv, "output"))
    {
      char *outputFilePath;
      getCmdLineArgumentString(argc, (const char **)argv, "output",
                               &outputFilePath);
      sResultFilename = outputFilePath;
    }
    // Host image object for an 8-bit grayscale image
    npp::ImageCPU_8u_C1 oHostSrc;
    // Load gray-scale image from the disk
    npp::loadImage(sFilename, oHostSrc);
    // declare a device image and copy construct from host image
    npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

    const int binCount = 255;
    const int levelCount = binCount + 1;

    // Assigning 32-bit signed integers for NPP representations
    Npp32s *histDevice = 0;
    Npp32s *levelsDevice = 0;

    NPP_CHECK_CUDA(cudaMalloc((void **)&histDevice, binCount * sizeof(Npp32s)));
    NPP_CHECK_CUDA(cudaMalloc((void **)&levelsDevice, levelCount * sizeof(Npp32s)));

    // Defining the Range of Interest
    NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};

    // Create Device-Scratch buffer for histogram
    int nDeviceBufferSize;
    // Computing device buffer size (in bytes) required by HistogramEven Primitives
    nppiHistogramEvenGetBufferSize_8u_C1R(oSizeROI, levelCount, &nDeviceBufferSize);
    // Pointer to the buffer created above
    Npp8u *pDeviceBuffer;
    NPP_CHECK_CUDA(cudaMalloc((void **)&pDeviceBuffer, nDeviceBufferSize));

    // Compute the levels values on host // LevelCount works as nLevel here
    Npp32s levelsHost[levelCount];
    NPP_CHECK_NPP(nppiEvenLevelsHost_32s(levelsHost, levelCount, 0, binCount));

    // Compute the histogram  // histDevice acts as *pHist here, oDeviceSrc.pitch() as nSrcStep(int)
    NPP_CHECK_NPP(nppiHistogramEven_8u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(), oSizeROI, histDevice, levelCount, 0, binCount, pDeviceBuffer));

    // copy histogram and levels to host memory
    Npp32s histHost[binCount];
    NPP_CHECK_CUDA(cudaMemcpy(histHost, histDevice, binCount * sizeof(Npp32s), cudaMemcpyDeviceToHost));

    Npp32s lutHost[levelCount];

    // Fill up look-up table
    {
      Npp32s *pHostHistogram = histHost;
      Npp32s totalSum = 0;

      for (; pHostHistogram < histHost + binCount; ++pHostHistogram)
      {
        totalSum += *pHostHistogram;
      }

      NPP_ASSERT(totalSum <= oSizeROI.width * oSizeROI.height);

      if (totalSum == 0)
      {
        totalSum = 1;
      }

      float multiplier = 1.0f / float(oSizeROI.width * oSizeROI.height) * 0xFF;

      Npp32s runningSum = 0;
      Npp32s *pLookupTable = lutHost;

      for (pHostHistogram = histHost; pHostHistogram < histHost + binCount; ++pHostHistogram)
      {
        *pLookupTable = (Npp32s)(runningSum * multiplier + 0.5f);
        pLookupTable++;
        runningSum += *pHostHistogram;
      }

      lutHost[binCount] = 0xFF; // last element is always 1
    }

    // Applying LUT transformation to the image

    // Create a device image for the result
    npp::ImageNPP_8u_C1 oDeviceDst(oDeviceSrc.size());

    // Perform image color processing using linear interpolation between various types of color look up tables
    NPP_CHECK_NPP(nppiLUT_Linear_8u_C1R(oDeviceSrc.data(), oDeviceSrc.pitch(), oDeviceDst.data(), oDeviceDst.pitch(), oSizeROI, lutHost, levelsHost, levelCount));

    npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

    cudaFree(histDevice);
    cudaFree(levelsDevice);
    cudaFree(pDeviceBuffer);
    nppiFree(oDeviceSrc.data());
    nppiFree(oDeviceDst.data());
    npp::saveImage(sResultFilename.c_str(), oHostDst);
    std::cout << "Saved image file " << sResultFilename << std::endl;
    exit(EXIT_SUCCESS);
  }
  catch (npp::Exception &rException)
  {
    std::cerr << "Program error! The following exception ocurred: \n";
    std::cerr << rException << std::endl;
    std::cerr << "Aborting..." << std::endl;
    exit(EXIT_FAILURE);
  }
  catch (...)
  {
    std::cerr << "Program error! An unknown exception occured. \n";
    std::cerr << "Aborting..." << std::endl;
    exit(EXIT_FAILURE);
  }

  return 0;
}