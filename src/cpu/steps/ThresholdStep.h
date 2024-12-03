#ifndef THRESHOLDSTEP_H
#define THRESHOLDSTEP_H

#include "Image.h"
#include "ProcessingStep.h"
#include <opencv2/imgproc.hpp>

class ThresholdStep : public ProcessingStep
{
 public:
    explicit ThresholdStep(int thresholdValue)
        : threshold(thresholdValue)
    {
    }

    void process(Image& img) override
    {
        // Convert to grayscale if the image is in color
        cv::Mat grayImg;
        if (img.getImage().channels() == 3)
        {
            cv::cvtColor(img.getImage(), grayImg, cv::COLOR_BGR2GRAY);
        }
        else
        {
            grayImg = img.getImage();
        }

        // Apply thresholding using Otsu's method
        cv::Mat binaryImg;
        cv::threshold(grayImg, binaryImg, threshold, 255,
                      cv::THRESH_BINARY | cv::THRESH_OTSU);

        // Set the processed image back
        img.setImage(binaryImg);
    }

 private:
    int threshold;
};

#endif  // THRESHOLDSTEP_H
