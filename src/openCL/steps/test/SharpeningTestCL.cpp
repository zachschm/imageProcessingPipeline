#include "Image.h"
#include "OpenCLManager.h"
#include "OpenCLSharpeningStep.h"
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

TEST(OpenCLSharpeningStep, SharpeningEffect)
{
    OpenCLManager openclManager;
    OpenCLSharpeningStep sharpeningStep(openclManager, 3);  // 3x3 kernel

    // Create a synthetic test image (100x100 grayscale gradient)
    cv::Mat testImage(100, 100, CV_8UC3, cv::Scalar(128));
    cv::rectangle(testImage, cv::Point(30, 30), cv::Point(70, 70),
                  cv::Scalar(200), -1);  // Bright square in the center

    Image img(testImage);

    // Process image
    sharpeningStep.process(img);

    // Validate output
    cv::Mat processedImage = img.getImage();

    ASSERT_EQ(processedImage.rows, testImage.rows);
    ASSERT_EQ(processedImage.cols, testImage.cols);

    // Ensure sharpening effect applied
    ASSERT_NE(cv::mean(processedImage), cv::mean(testImage));

    // Check that the center bright region remains prominent
    ASSERT_GT(processedImage.at<uchar>(50, 50),
              processedImage.at<uchar>(20, 20));
}
