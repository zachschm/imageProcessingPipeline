#ifndef SATURATIONSTEP_H
#define SATURATIONSTEP_H

#include "Image.h"
#include "ProcessingStep.h"
#include <opencv2/imgproc.hpp>

class SaturationStep : public ProcessingStep
{
 public:
    SaturationStep(double scale)
        : scale(scale)
    {
        // Ensure the scale is within a reasonable range
        if (scale < 0)
        {
            scale = 0;  // Avoid negative scaling
        }
        if (scale > 3)
        {
            scale = 3;  // Cap scaling to a maximum of 3x
        }
    }

    void process(Image& img) override
    {
        // Convert the image to HSV color space
        cv::Mat hsvImage;
        cv::cvtColor(img.getImage(), hsvImage, cv::COLOR_BGR2HSV);

        // Split into individual channels (Hue, Saturation, Value)
        std::vector<cv::Mat> channels;
        cv::split(hsvImage, channels);

        // Adjust the saturation channel
        for (int i = 0; i < channels[1].rows; ++i)
        {
            for (int j = 0; j < channels[1].cols; ++j)
            {
                uchar originalSaturation = channels[1].at<uchar>(i, j);
                uchar newSaturation =
                    cv::saturate_cast<uchar>(originalSaturation * scale);
                channels[1].at<uchar>(i, j) = newSaturation;
            }
        }

        // Merge the channels back together
        cv::merge(channels, hsvImage);

        // Convert back to BGR color space
        cv::cvtColor(hsvImage, img.getImage(), cv::COLOR_HSV2BGR);
    }

 private:
    double scale;  // Scale factor for saturation adjustment
};

#endif  // SATURATIONSTEP_H
