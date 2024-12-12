#include "GrayscaleStepCL.h"

GrayscaleStepCL::GrayscaleStepCL(OpenCLManager& manager)
    : openclManager(manager)
{
    std::string kernelPath = std::string(std::getenv("PROJECT_ROOT")) + "/src/openCL/kernels/grayscale.cl";
    openclManager.loadKernel("grayscale", kernelPath);
    kernel = openclManager.getKernel("grayscale");
}

void GrayscaleStepCL::process(Image& img)
{
    cv::Mat inputImage = img.getImage();
    if (inputImage.empty() || inputImage.channels() != 3)
    {
        throw std::runtime_error("Input image must be a non-empty RGB image.");
    }

    int width = inputImage.cols;
    int height = inputImage.rows;

    // Create OpenCL buffers
    cl::Buffer inputBuffer(
        openclManager.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        inputImage.total() * inputImage.elemSize(), inputImage.data);
    cl::Buffer outputBuffer(openclManager.getContext(), CL_MEM_WRITE_ONLY,
                            width * height * sizeof(unsigned char));

    // Set kernel arguments
    kernel.setArg(0, inputBuffer);
    kernel.setArg(1, outputBuffer);
    kernel.setArg(2, width);
    kernel.setArg(3, height);

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

    // Update the image with the processed grayscale data
    img.setImage(outputImage);
}
