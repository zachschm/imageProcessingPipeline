#ifndef HISTOGRAMEQUALIZATIONSTEPCL_H
#define HISTOGRAMEQUALIZATIONSTEPCL_H

#include "Image.h"
#include "OpenCLManager.h"
#include "ProcessingStep.h"
#include <opencv2/opencv.hpp>
#include <vector>

class HistogramEqualizationStepCL : public ProcessingStep
{
 public:
    explicit HistogramEqualizationStepCL(OpenCLManager& manager);

    void process(Image& img) override;

 private:
    OpenCLManager& openclManager;
    cl::Kernel kernel;

    std::vector<cv::Mat> splitImage(const cv::Mat& img, int parts);
    cv::Mat mergeImage(const std::vector<std::vector<uchar>>& segments,
                       int width, int height);
};

#endif  // HISTOGRAMEQUALIZATIONSTEPCL_H
