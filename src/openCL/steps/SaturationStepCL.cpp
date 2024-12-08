#include "SaturationStepCL.h"

SaturationStepCL::SaturationStepCL(OpenCLManager& manager, float scale)
    : manager(manager)
    , scale(scale)
{
    manager.loadKernel("adjustSaturation", "src/opencl/kernels/saturation.cl");
    kernel = manager.getKernel("adjustSaturation");
}

void SaturationStepCL::process(Image& img)
{
    cv::Mat inputImage = img.getImage();

    if (inputImage.empty())
    {
        throw std::runtime_error("Input image is empty.");
    }

    // Convert image to HSV
    cv::Mat hsvImage;
    cv::cvtColor(inputImage, hsvImage, cv::COLOR_BGR2HSV);

    // Split image for multi-GPU processing
    auto subImages = ImageSplitter::split(hsvImage, manager.getDeviceCount());

    std::vector<Image> processedSubImages;  // Changed to store `Image` objects

#pragma omp parallel for
    for (int i = 0; i < subImages.size(); ++i)
    {
        cl::Image2D inputBuffer =
            manager.createImage2DFromMat(subImages[i].getImage(), i);
        cl::Image2D outputBuffer(manager.getContext(), CL_MEM_WRITE_ONLY,
                                 cl::ImageFormat(CL_RGBA, CL_UNORM_INT8),
                                 subImages[i].getCols(),
                                 subImages[i].getRows());

        // Set kernel arguments
        kernel.setArg(0, inputBuffer);
        kernel.setArg(1, outputBuffer);
        kernel.setArg(2, scale);

        // Execute kernel
        cl::CommandQueue& queue = manager.getQueue(i);
        cl::NDRange globalSize(subImages[i].getCols(), subImages[i].getRows());
        queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize);
        queue.finish();

        // Retrieve processed sub-image
        cv::Mat processedPart = manager.readImage2DToMat(
            outputBuffer, subImages[i].getCols(), subImages[i].getRows(), i);

#pragma omp critical
        processedSubImages.emplace_back(
            processedPart);  // Wrap `cv::Mat` in `Image`
    }

    // Merge processed sub-images
    Image mergedImage = ImageMerger::merge(processedSubImages);

    // Convert back to BGR
    cv::cvtColor(mergedImage.getImage(), img.getImage(), cv::COLOR_HSV2BGR);
}
