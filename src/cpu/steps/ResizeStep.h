#ifndef RESIZESTEP_H
#define RESIZESTEP_H

#include "Image.h"
#include "Pipeline.h"
#include <opencv2/opencv.hpp>

class ResizeStep : public ProcessingStep
{
 public:
    ResizeStep(int w = 1920, int h = 1080)
        : width(w)
        , height(h)
    {
    }

    void process(Image& img) override
    {
        if (!img.getImage().empty())
        {
            cv::Mat resizedImg;
            cv::resize(img.getImage(), resizedImg, cv::Size(width, height));
            img.setImage(resizedImg);
        }
    }

 private:
    int width;
    int height;
};

#endif
