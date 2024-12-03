#ifndef SOBELEDGESTEP_H
#define SOBELEDGESTEP_H

#include "Image.h"
#include "ProcessingStep.h"
#include <opencv2/opencv.hpp>

class SobelEdgeStep : public ProcessingStep
{
 public:
    void process(Image& img) override
    {
        if (!img.getImage().empty())
        {
            cv::Mat grayImage, gradX, gradY, absGradX, absGradY,
                gradientMagnitude;

            // Check if the image is already grayscale
            if (img.getImage().channels() == 3)
            {
                // Convert to grayscale if the image has 3 channels (BGR)
                cv::cvtColor(img.getImage(), grayImage, cv::COLOR_BGR2GRAY);
            }
            else
            {
                // If already grayscale, just use the image directly
                grayImage = img.getImage();
            }

            // Calculate gradients
            cv::Sobel(grayImage, gradX, CV_16S, 1, 0, 3);
            cv::Sobel(grayImage, gradY, CV_16S, 0, 1, 3);

            // Convert to absolute values
            cv::convertScaleAbs(gradX, absGradX);
            cv::convertScaleAbs(gradY, absGradY);

            // Combine the two gradients
            cv::addWeighted(absGradX, 0.5, absGradY, 0.5, 0, gradientMagnitude);

            // Set the processed image
            img.setImage(gradientMagnitude);
        }
    }
};

#endif  // SOBELEDGESTEP_H
