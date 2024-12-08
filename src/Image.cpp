#include "Image.h"
#include <opencv2/opencv.hpp>
#include <string>

Image::Image() = default;

Image::Image(const cv::Mat& img)
{
    setImage(img);
}

bool Image::load(const std::string& path)
{
    image = cv::imread(path);
    return !image.empty();
}

bool Image::save(const std::string& outputPath) const
{
    return cv::imwrite(outputPath, image);
}

cv::Mat& Image::getImage()
{
    return image;
}

const cv::Mat& Image::getImage() const  // Const version
{
    return image;
}

void Image::setImage(const cv::Mat& img)
{
    image = img;
}

const int Image::getRows() const
{
    return image.rows;
}

const int Image::getCols() const
{
    return image.cols;
}
