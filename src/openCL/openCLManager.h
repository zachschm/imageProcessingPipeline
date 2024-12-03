#ifndef OPENCLMANAGER_H
#define OPENCLMANAGER_H

#include <CL/cl.hpp>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

struct GPUContext
{
    cl::Context context;
    cl::CommandQueue queue;
    cl::Device device;
};

class OpenCLManager
{
 public:
    OpenCLManager();
    ~OpenCLManager();

    void initialize();
    cl::Buffer createPinnedBuffer(size_t size, int gpuIndex);
    cl::Buffer getReusableBuffer(size_t size, int gpuIndex);
    cl::CommandQueue& getQueue(int gpuIndex);
    cl::Device& getDevice(int gpuIndex);
    int getDeviceCount() const;
    void cleanup();

 private:
    std::vector<GPUContext> gpuContexts;
    std::vector<std::unordered_map<size_t, cl::Buffer>> bufferPools;

    void initializeBufferPools();
    std::string readKernelFile(const std::string& filePath) const;
};

#endif  // OPENCLMANAGER_H
