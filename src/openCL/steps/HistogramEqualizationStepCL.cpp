#include "HistogramEqualizationStepCL.h"
#include <stdexcept>

HistogramEqualizationStepCL::HistogramEqualizationStepCL(OpenCLManager& manager)
    : openclManager(manager)
{
    // Load the histogram equalization kernel
    openclManager.loadKernel("histogram_equalization",
                             "src/opencl/kernels/histogram_equalization.cl");
    kernel = openclManager.getKernel("histogram_equalization");
}

void HistogramEqualizationStepCL::process(Image& img)
{
    cv::Mat inputImage = img.getImage();
    if (inputImage.empty() || inputImage.channels() != 1)
    {
        throw std::runtime_error(
            "Input image must be a non-empty grayscale image.");
    }

    int width = inputImage.cols;
    int height = inputImage.rows;

    // Create OpenCL buffers
    cl::Buffer inputBuffer(
        openclManager.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        inputImage.total() * inputImage.elemSize(), inputImage.data);
    cl::Buffer outputBuffer(openclManager.getContext(), CL_MEM_WRITE_ONLY,
                            width * height * sizeof(unsigned char));
    cl::Buffer histBuffer(openclManager.getContext(), CL_MEM_READ_WRITE,
                          sizeof(int) * 256, nullptr);
    cl::Buffer cdfBuffer(openclManager.getContext(), CL_MEM_READ_WRITE,
                         sizeof(int) * 256, nullptr);

    // Set kernel arguments
    kernel.setArg(0, inputBuffer);
    kernel.setArg(1, outputBuffer);
    kernel.setArg(2, histBuffer);
    kernel.setArg(3, cdfBuffer);
    kernel.setArg(4, width);
    kernel.setArg(5, height);

    // Execute kernel
    cl::CommandQueue& queue =
        openclManager.getQueue(0 /* Assume GPU is pre-selected */);
    queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                               cl::NDRange(width, height));
    queue.finish();

    // Retrieve processed data
    cv::Mat outputImage(height, width, CV_8UC1);
    queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, outputImage.total(),
                            outputImage.data);

    // Update the image with the processed histogram-equalized data
    img.setImage(outputImage);
}
