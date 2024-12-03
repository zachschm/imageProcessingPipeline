#ifndef SHARPENINGSTEP_H
#define SHARPENINGSTEP_H

#include "Image.h"
#include "ProcessingStep.h"
#include <opencv2/imgproc.hpp>

class SharpeningStep : public ProcessingStep
{
 public:
    void process(Image& img) override
    {
        // Create a sharpening kernel
        cv::Mat kernel =
            (cv::Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);

        // Apply the sharpening filter
        cv::filter2D(img.getImage(), img.getImage(), img.getImage().depth(),
                     kernel);
    }
};

#endif  // SHARPENINGSTEP_H
