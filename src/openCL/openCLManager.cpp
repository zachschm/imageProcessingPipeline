#include "OpenCLManager.h"
#include <fstream>
#include <sstream>

OpenCLManager::OpenCLManager() = default;

OpenCLManager::~OpenCLManager()
{
    cleanup();
}

void OpenCLManager::initialize()
{
    try
    {
        // Get all platforms
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty())
        {
            throw std::runtime_error("No OpenCL platforms found.");
        }

        // Select the first platform
        cl::Platform platform = platforms.front();

        // Get all GPU devices for the platform
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty())
        {
            throw std::runtime_error("No GPU devices found on the platform.");
        }

        if (devices.size() < 4)
        {
            throw std::runtime_error(
                "Expected at least 4 GPUs for G4dn instances.");
        }

        // Create GPU contexts for the first 4 GPUs
        for (int i = 0; i < 4; ++i)
        {
            GPUContext gpuContext;
            gpuContext.device = devices[i];
            gpuContext.context = cl::Context(gpuContext.device);
            gpuContext.queue =
                cl::CommandQueue(gpuContext.context, gpuContext.device);
            gpuContexts.push_back(gpuContext);

            std::cout << "Initialized GPU " << i << ": "
                      << gpuContext.device.getInfo<CL_DEVICE_NAME>() << "\n";
        }

        initializeBufferPools();
    }
    catch (const cl::Error& e)
    {
        throw std::runtime_error("OpenCL initialization failed: " +
                                 std::string(e.what()));
    }
}

void OpenCLManager::initializeBufferPools()
{
    bufferPools.resize(gpuContexts.size());
}

cl::Buffer OpenCLManager::createPinnedBuffer(size_t size, int gpuIndex)
{
    if (gpuIndex < 0 || gpuIndex >= gpuContexts.size())
    {
        throw std::out_of_range("Invalid GPU index.");
    }
    return cl::Buffer(gpuContexts[gpuIndex].context,
                      CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size);
}

cl::Buffer OpenCLManager::getReusableBuffer(size_t size, int gpuIndex)
{
    if (gpuIndex < 0 || gpuIndex >= gpuContexts.size())
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
    if (gpuIndex < 0 || gpuIndex >= gpuContexts.size())
    {
        throw std::out_of_range("Invalid GPU index.");
    }
    return gpuContexts[gpuIndex].queue;
}

cl::Device& OpenCLManager::getDevice(int gpuIndex)
{
    if (gpuIndex < 0 || gpuIndex >= gpuContexts.size())
    {
        throw std::out_of_range("Invalid GPU index.");
    }
    return gpuContexts[gpuIndex].device;
}

int OpenCLManager::getDeviceCount() const
{
    return gpuContexts.size();
}

void OpenCLManager::cleanup()
{
    gpuContexts.clear();
    bufferPools.clear();
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
