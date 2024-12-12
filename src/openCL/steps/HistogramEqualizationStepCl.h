#ifndef HISTOGRAMEQUALIZATIONSTEPCL_H
#define HISTOGRAMEQUALIZATIONSTEPCL_H

#include "Image.h"
#include "OpenCLManager.h"
#include "ProcessingStep.h"
#include <opencv2/opencv.hpp>

class HistogramEqualizationStepCL : public ProcessingStep
{
 public:
    explicit HistogramEqualizationStepCL(OpenCLManager& manager);

    void process(Image& img) override;

 private:
    OpenCLManager& openclManager;
    cl::Kernel kernel;
};

#endif  // HISTOGRAMEQUALIZATIONSTEPCL_H
