#include "Image.h"
#include "OpenCLGaussianBlurStep.h"
#include "OpenCLManager.h"
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

// This main function runs all tests.
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

TEST(OpenCLGaussianBlurStep, BlurEffect)
{
    OpenCLManager openclManager;
    OpenCLGaussianBlurStep blurStep(openclManager, 5, 1.0f);

    // Create a synthetic test image
    cv::Mat testImage(100, 100, CV_8UC1, cv::Scalar(128));
    cv::rectangle(testImage, cv::Point(30, 30), cv::Point(70, 70),
                  cv::Scalar(200), -1);

    Image img(testImage);

    // Process image
    blurStep.process(img);

    // Validate output
    cv::Mat processedImage = img.getImage();
    ASSERT_EQ(processedImage.rows, testImage.rows);
    ASSERT_EQ(processedImage.cols, testImage.cols);

    // Ensure that the processed image is different from the original
    ASSERT_NE(cv::mean(testImage)[0], cv::mean(processedImage)[0]);
}
