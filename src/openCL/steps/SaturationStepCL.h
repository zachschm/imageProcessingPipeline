#ifndef SATURATIONSTEPCL_H
#define SATURATIONSTEPCL_H

#include "Image.h"
#include "OpenCLManager.h"
#include "ProcessingStep.h"
#include <opencv2/opencv.hpp>

class SaturationStepCL : public ProcessingStep
{
 public:
    SaturationStepCL(OpenCLManager& manager, float scale);
    void process(Image& img) override;

 private:
    OpenCLManager& manager;
    cl::Kernel kernel;
    float scale;
};

#endif  // SATURATIONSTEPCL_H
