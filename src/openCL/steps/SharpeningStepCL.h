#ifndef SHARPENINGSTEPCL_H
#define SHARPENINGSTEPCL_H

#include "Image.h"
#include "OpenCLManager.h"
#include "ProcessingStep.h"
#include <vector>

class SharpeningStepCL : public ProcessingStep
{
 public:
    SharpeningStepCL(OpenCLManager& manager, int kernelSize);
    void process(Image& img) override;

 private:
    OpenCLManager& manager;
    cl::Kernel kernel;
    int kernelSize;
    std::vector<float> sharpeningKernel;
};

#endif  // SHARPENINGSTEPCL_H
