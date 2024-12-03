#ifndef IMAGEMERGER_H
#define IMAGEMERGER_H

#include "Image.h"
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <vector>

class ImageMerger
{
 public:
    // Merge a vector of image parts into a single image
    static Image merge(const std::vector<Image>& parts)
    {
        if (parts.empty())
        {
            throw std::invalid_argument("No image parts provided for merging.");
        }

        // Validate that all parts have the same height
        int partHeight = parts[0].getImage().rows;
        for (const auto& part : parts)
        {
            if (part.getImage().rows != partHeight)
            {
                throw std::runtime_error(
                    "All image parts must have the same height.");
            }
        }

        // Merge the parts horizontally
        cv::Mat mergedImage;
        std::vector<cv::Mat> partMats;
        for (const auto& part : parts)
        {
            partMats.push_back(part.getImage());
        }

        cv::hconcat(partMats, mergedImage);

        return Image(mergedImage);
    }
};

#endif  // IMAGEMERGER_H
