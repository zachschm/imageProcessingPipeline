#ifndef IMAGESPLITTER_H
#define IMAGESPLITTER_H

#include "Image.h"
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <vector>

class ImageSplitter
{
 public:
    // Split an image into `numParts` vertical strips
    static std::vector<Image> split(const cv::Mat& img, int numParts)
    {
        if (numParts <= 0)
        {
            throw std::invalid_argument(
                "Number of parts must be greater than 0.");
        }

        cv::Mat original = img;
        int width = original.cols;
        int height = original.rows;

        if (width % numParts != 0)
        {
            throw std::runtime_error(
                "Image width must be divisible by the number of parts.");
        }

        int partWidth = width / numParts;
        std::vector<Image> parts;

        for (int i = 0; i < numParts; ++i)
        {
            // Define the ROI for each part
            cv::Rect roi(i * partWidth, 0, partWidth, height);
            cv::Mat partMat = original(roi);

            // Wrap the ROI in an Image object
            parts.emplace_back(Image(partMat.clone()));
        }

        return parts;
    }
};

#endif  // IMAGESPLITTER_H
