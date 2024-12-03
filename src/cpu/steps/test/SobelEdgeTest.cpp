#include "Image.h"
#include "SobelEdgeStep.h"
#include <gtest/gtest.h>

// Test case for Sobel Edge Detection
// Test case for Sobel Edge Detection
TEST(SobelEdgeTest, EdgeDetectionEffectIsAppliedCorrectly)
{
    // Create a simple image object (100x100 with a white square in the center)
    cv::Mat originalImage(100, 100, CV_8UC1, cv::Scalar(0));  // Black image
    cv::rectangle(originalImage, cv::Point(30, 30), cv::Point(70, 70),
                  cv::Scalar(255), -1);  // White square
    Image img(originalImage);

    // Create the SobelEdgeStep
    SobelEdgeStep sobelEdgeStep;

    // Apply Sobel edge detection
    sobelEdgeStep.process(img);

    // Get the processed image
    cv::Mat edgeImage = img.getImage();

    // Check that some pixels in the center region are white (indicating edges)
    ASSERT_GT(edgeImage.at<uchar>(50, 30), 0);  // Left edge
    ASSERT_GT(edgeImage.at<uchar>(30, 50), 0);  // Top edge
    ASSERT_GT(edgeImage.at<uchar>(50, 70), 0);  // Right edge
    ASSERT_GT(edgeImage.at<uchar>(70, 50), 0);  // Bottom edge
}

// Test case for no edges detected
TEST(SobelEdgeTest, NoEdgeDetectionOnUniformImage)
{
    // Create a simple uniform image (100x100 with a single color)
    cv::Mat originalImage(100, 100, CV_8UC1, cv::Scalar(128));  // Gray image
    Image img(originalImage);

    // Create the SobelEdgeStep
    SobelEdgeStep sobelEdgeStep;

    // Apply Sobel edge detection
    sobelEdgeStep.process(img);

    // Get the processed image
    cv::Mat edgeImage = img.getImage();

    // Check that all pixels are zero (no edges detected)
    for (int i = 0; i < edgeImage.rows; ++i)
    {
        for (int j = 0; j < edgeImage.cols; ++j)
        {
            ASSERT_EQ(edgeImage.at<uchar>(i, j), 0);
        }
    }
}
