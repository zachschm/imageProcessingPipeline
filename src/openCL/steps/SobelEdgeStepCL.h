#ifndef SOBELEDGESTEPCL_H
#define SOBELEDGESTEPCL_H

#include "Image.h"
#include "ImageMerger.h"
#include "ImageSplitter.h"
#include "OpenCLManager.h"
#include "ProcessingStep.h"
#include <vector>

class SobelEdgeStepCl : public ProcessingStep
{
 public:
    explicit SobelEdgeStepCl(OpenCLManager& manager);
    void process(Image& img) override;

 private:
    OpenCLManager& manager;
    cl::Kernel kernel;
};

#endif  // SOBELEDGESTEPCL_H
