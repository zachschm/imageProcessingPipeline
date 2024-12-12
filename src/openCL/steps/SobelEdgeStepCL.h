#ifndef SOBELEDGESTEPCL_H
#define SOBELEDGESTEPCL_H

#include "Image.h"
#include "OpenCLManager.h"
#include "ProcessingStep.h"

class SobelEdgeStepCL : public ProcessingStep
{
 public:
    explicit SobelEdgeStepCL(OpenCLManager& manager);
    void process(Image& img) override;

 private:
    OpenCLManager& manager;
    cl::Kernel kernel;
};

#endif  // SOBELEDGESTEPCL_H
