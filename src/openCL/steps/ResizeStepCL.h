#ifndef RESIZESTEPCL_H
#define RESIZESTEPCL_H

#include "Image.h"
#include "OpenCLManager.h"
#include "ProcessingStep.h"
#include <opencv2/opencv.hpp>

class ResizeStepCL : public ProcessingStep
{
 public:
    ResizeStepCL(OpenCLManager& manager, int newWidth, int newHeight);
    void process(Image& img) override;

 private:
    OpenCLManager& manager;
    int newWidth;
    int newHeight;
    cl::Kernel kernel;
};

#endif  // RESIZESTEPCL_H
