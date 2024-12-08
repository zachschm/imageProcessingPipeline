#ifndef IMAGE_H
#define IMAGE_H

#include <opencv2/opencv.hpp>
#include <string>

class Image
{
 public:
    Image();
    explicit Image(const cv::Mat& img);

    bool load(const std::string& path);
    bool save(const std::string& outputPath) const;

    cv::Mat& getImage();
    const cv::Mat& getImage() const;
    void setImage(const cv::Mat& img);
    const int getRows() const;
    const int getCols() const;

 private:
    cv::Mat image;
};

#endif  // IMAGE_H
