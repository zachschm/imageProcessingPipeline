#include "OpenCLManager.h"
#include <array>
#include <fstream>
#include <iostream>
#include <sstream>

OpenCLManager::OpenCLManager()
{
    try
    {
        initialize();
    }
    catch (const std::exception& e)
    {
        std::cerr << "Failed to initialize OpenCLManager: " << e.what()
                  << std::endl;
        throw;
    }
}

OpenCLManager::~OpenCLManager()
{
    cleanup();
}

void OpenCLManager::initialize()
{
    // Retrieve platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty())
    {
        throw std::runtime_error("No OpenCL platforms found.");
    }

    // Select the NVIDIA CUDA platform
    cl::Platform selectedPlatform;
    for (const auto& p : platforms)
    {
        std::string platformName = p.getInfo<CL_PLATFORM_NAME>();
        if (platformName.find("NVIDIA CUDA") != std::string::npos)
        {
            selectedPlatform = p;
            break;
        }
    }

    if (selectedPlatform() == nullptr)
    {
        throw std::runtime_error("NVIDIA CUDA platform not found.");
    }

    std::cout << "Selected platform: " << selectedPlatform.getInfo<CL_PLATFORM_NAME>() << std::endl;

    // Get GPU devices
    std::vector<cl::Device> devices;
    selectedPlatform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty())
    {
        throw std::runtime_error("No GPU devices found on the selected platform.");
    }

    std::cout << "Available devices: " << devices.size() << std::endl;
    for (const auto& device : devices)
    {
        std::cout << "Device: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
    }

    // Ensure at least 4 GPUs are available
    if (devices.size() < 4)
    {
        throw std::runtime_error("Less than 4 GPUs detected.");
    }

    // Create a unified context
    std::vector<cl::Device> chosenDevices(devices.begin(), devices.begin() + 4);
    unifiedContext = cl::Context(chosenDevices);

    // Create command queues
    gpuContexts.clear();
    for (int i = 0; i < 4; ++i)
    {
        GPUContext gpuContext;
        gpuContext.device = chosenDevices[i];
        gpuContext.queue = cl::CommandQueue(unifiedContext, gpuContext.device);
        gpuContexts.push_back(gpuContext);

        std::cout << "Initialized GPU " << i << ": "
                  << gpuContext.device.getInfo<CL_DEVICE_NAME>() << "\n";
    }

    initializeBufferPools();
}

void OpenCLManager::initializeBufferPools()
{
    bufferPools.resize(gpuContexts.size());
}

cl::Buffer OpenCLManager::createPinnedBuffer(size_t size, int gpuIndex)
{
    if (gpuIndex < 0 || gpuIndex >= (int)gpuContexts.size())
    {
        throw std::out_of_range("Invalid GPU index.");
    }
    return cl::Buffer(unifiedContext, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                      size);
}

cl::Buffer OpenCLManager::getReusableBuffer(size_t size, int gpuIndex)
{
    if (gpuIndex < 0 || gpuIndex >= (int)gpuContexts.size())
    {
        throw std::out_of_range("Invalid GPU index.");
    }

    auto& pool = bufferPools[gpuIndex];
    auto it = pool.find(size);
    if (it != pool.end())
    {
        return it->second;
    }

    cl::Buffer newBuffer = createPinnedBuffer(size, gpuIndex);
    pool[size] = newBuffer;
    return newBuffer;
}

cl::CommandQueue& OpenCLManager::getQueue(int gpuIndex)
{
    if (gpuIndex < 0 || gpuIndex >= (int)gpuContexts.size())
    {
        throw std::out_of_range("Invalid GPU index.");
    }
    return gpuContexts[gpuIndex].queue;
}

cl::Device& OpenCLManager::getDevice(int gpuIndex)
{
    if (gpuIndex < 0 || gpuIndex >= (int)gpuContexts.size())
    {
        throw std::out_of_range("Invalid GPU index.");
    }
    return gpuContexts[gpuIndex].device;
}

int OpenCLManager::getDeviceCount() const
{
    return (int)gpuContexts.size();
}

void OpenCLManager::cleanup()
{
    gpuContexts.clear();
    bufferPools.clear();
    kernels.clear();
}

std::string OpenCLManager::readKernelFile(const std::string& filePath) const
{
    std::ifstream file(filePath);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open kernel file: " + filePath);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// New methods for kernel management
void OpenCLManager::loadKernel(const std::string& kernelName,
                               const std::string& fileName)
{
    // Determine the kernel file path
    std::string kernelFile =
        fileName.empty() ? kernelDir + kernelName + ".cl" : fileName;

    std::string source = readKernelFile(kernelFile);
    cl::Program::Sources sources;
    sources.push_back({source.c_str(), source.length()});

    cl::Program program(unifiedContext, sources);
    // Build program for all devices in unifiedContext
    program.build();

    // Create the kernel
    cl::Kernel kernel(program, kernelName.c_str());
    kernels[kernelName] = kernel;
}

cl::Kernel OpenCLManager::getKernel(const std::string& kernelName)
{
    auto it = kernels.find(kernelName);
    if (it == kernels.end())
    {
        throw std::runtime_error("Kernel not found: " + kernelName);
    }
    return it->second;
}

// Image-related methods

cl::Image2D OpenCLManager::createImage2DFromMat(const cv::Mat& mat,
                                                int gpuIndex)
{
    if (gpuIndex < 0 || gpuIndex >= (int)gpuContexts.size())
    {
        throw std::out_of_range("Invalid GPU index.");
    }

    cv::Mat floatMat;
    if (mat.type() != CV_32FC1)
    {
        mat.convertTo(floatMat, CV_32FC1);
    }
    else
    {
        floatMat = mat;
    }

    int width = floatMat.cols;
    int height = floatMat.rows;

    cl::ImageFormat format(CL_R, CL_FLOAT);
    cl::Image2D image(unifiedContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                      format, width, height, 0, (void*)floatMat.data);

    return image;
}

cv::Mat OpenCLManager::readImage2DToMat(const cl::Image2D& image, int width,
                                        int height, int gpuIndex)
{
    if (gpuIndex < 0 || gpuIndex >= (int)gpuContexts.size())
    {
        throw std::out_of_range("Invalid GPU index.");
    }

    cv::Mat outMat(height, width, CV_32FC1);

    std::array<size_t, 2> originArr = {0, 0};
    std::array<size_t, 2> regionArr = {(size_t)width, (size_t)height};

    gpuContexts[gpuIndex].queue.enqueueReadImage(image, CL_TRUE, originArr,
                                                 regionArr,
                                                 0,  // row pitch
                                                 0,  // slice pitch
                                                 outMat.data);

    return outMat;
}
