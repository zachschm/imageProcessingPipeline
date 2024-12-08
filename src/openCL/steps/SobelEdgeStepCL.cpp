#include "SobelEdgeStepCl.h"
#include <opencv2/opencv.hpp>
#include <stdexcept>

SobelEdgeStepCl::SobelEdgeStepCl(OpenCLManager& manager)
    : manager(manager)
{
    manager.loadKernel("sobelEdge", "src/opencl/kernels/sobel_edge.cl");
    kernel = manager.getKernel("sobelEdge");
}

void SobelEdgeStepCl::process(Image& img)
{
    cv::Mat inputImage = img.getImage();
    if (inputImage.empty())
    {
        throw std::runtime_error("Input image is empty.");
    }

    // Convert image to grayscale
    cv::Mat grayscaleImage;
    cv::cvtColor(inputImage, grayscaleImage, cv::COLOR_BGR2GRAY);

    // Split the image for multi-GPU processing
    auto subImages =
        ImageSplitter::split(grayscaleImage, manager.getDeviceCount());
    std::vector<Image> processedSubImages;  // Changed to store `Image` objects

#pragma omp parallel for
    for (int i = 0; i < subImages.size(); ++i)
    {
        cl::Image2D inputBuffer =
            manager.createImage2DFromMat(subImages[i].getImage(), i);
        cl::Image2D outputBuffer(manager.getContext(), CL_MEM_WRITE_ONLY,
                                 cl::ImageFormat(CL_R, CL_FLOAT),
                                 subImages[i].getCols(),
                                 subImages[i].getRows());

        // Set kernel arguments
        kernel.setArg(0, inputBuffer);
        kernel.setArg(1, outputBuffer);

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

    // Set the processed image
    img.setImage(mergedImage.getImage());
}
