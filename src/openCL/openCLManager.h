#ifndef OPENCLMANAGER_H
#define OPENCLMANAGER_H
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_ENABLE_EXCEPTIONS

#include <CL/opencl.hpp>
#include <map>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <vector>

class OpenCLManager
{
 public:
    OpenCLManager();
    ~OpenCLManager();

    void initialize();
    int getDeviceCount() const;
    cl::CommandQueue& getQueue(int gpuIndex);
    cl::Device& getDevice(int gpuIndex);
    cl::Context& getContext()
    {
        return unifiedContext;
    }
    void cleanup();

    // New methods for kernel management
    void loadKernel(const std::string& kernelName,
                    const std::string& fileName = "");
    cl::Kernel getKernel(const std::string& kernelName);

    // Methods required by steps
    cl::Image2D createImage2DFromMat(const cv::Mat& mat, int gpuIndex);
    cv::Mat readImage2DToMat(const cl::Image2D& image, int width, int height,
                             int gpuIndex);

 private:
    struct GPUContext
    {
        cl::Device device;
        cl::CommandQueue queue;
    };

    std::vector<GPUContext> gpuContexts;
    std::vector<std::map<size_t, cl::Buffer>> bufferPools;

    cl::Context unifiedContext;  // Single context for all devices
    std::map<std::string, cl::Kernel> kernels;

    void initializeBufferPools();
    cl::Buffer createPinnedBuffer(size_t size, int gpuIndex);
    cl::Buffer getReusableBuffer(size_t size, int gpuIndex);
    std::string readKernelFile(const std::string& filePath) const;

    std::string kernelDir = "src/opencl/kernels/";
};

#endif  // OPENCLMANAGER_H