#ifndef GRAYSCALESTEPCL_H
#define GRAYSCALESTEPCL_H

#include "Image.h"
#include "OpenCLManager.h"
#include "ProcessingStep.h"
#include <opencv2/opencv.hpp>
#include <vector>

class GrayscaleStepCL : public ProcessingStep
{
 public:
    explicit GrayscaleStepCL(OpenCLManager& manager);

    void process(Image& img) override;

 private:
    OpenCLManager& openclManager;
    cl::Kernel kernel;
};

#endif  // GRAYSCALESTEPCL_H
