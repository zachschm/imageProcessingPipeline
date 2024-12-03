#ifndef GAUSSIANBLURSTEPCL_H
#define GAUSSIANBLURSTEPCL_H

#include "Image.h"
#include "ImageMerger.h"
#include "ImageSplitter.h"
#include "OpenCLManager.h"
#include "ProcessingStep.h"
#include <opencv2/opencv.hpp>
#include <vector>

class GaussianBlurStepCL : public ProcessingStep
{
 public:
    GaussianBlurStepCL(OpenCLManager& manager, int kernelSize, float sigma);
    void process(Image& img) override;

 private:
    OpenCLManager& openclManager;
    cl::Kernel kernel;
    int kernelSize;
    float sigma;
    std::vector<float> gaussianKernel;

    void precomputeKernel();
};

#endif  // GAUSSIANBLURSTEPCL_H
