#ifndef GRAYSCALESTEP_H
#define GRAYSCALESTEP_H

#include "Image.h"
#include "Pipeline.h"

class GrayscaleStep : public ProcessingStep
{
 public:
    void process(Image& img) override
    {
        if (!img.getImage().empty())
        {
            cv::Mat grayscaleImg;
            cv::cvtColor(img.getImage(), grayscaleImg, cv::COLOR_BGR2GRAY);
            img.setImage(grayscaleImg);
        }
    }
};

#endif
