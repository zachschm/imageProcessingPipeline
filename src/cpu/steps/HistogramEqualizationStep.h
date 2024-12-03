#ifndef HISTOGRAMEQUALIZATIONSTEP_H
#define HISTOGRAMEQUALIZATIONSTEP_H

#include "Image.h"
#include "ProcessingStep.h"
#include <opencv2/imgproc.hpp>

class HistogramEqualizationStep : public ProcessingStep
{
 public:
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

        // Apply histogram equalization
        cv::Mat equalizedImg;
        cv::equalizeHist(grayImg, equalizedImg);

        // Set the processed image back
        img.setImage(equalizedImg);
    }
};

#endif  // HISTOGRAMEQUALIZATIONSTEP_H
