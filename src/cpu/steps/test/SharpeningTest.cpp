#include "Image.h"
#include "SharpeningStep.h"
#include <gtest/gtest.h>

// TEST(SharpeningTest, SharpeningEffectOnPredefinedShapes)
// {
//     // Create a black image (100x100)
//     cv::Mat originalImage = cv::Mat::zeros(100, 100, CV_8UC3);

//     // Draw a white circle in the center
//     cv::circle(originalImage, cv::Point(50, 50), 30, cv::Scalar(255, 255,
//     255),
//                cv::FILLED);

//     // Draw a white line
//     cv::line(originalImage, cv::Point(0, 0), cv::Point(100, 100),
//              cv::Scalar(255, 255, 255), 2);

//     Image img(originalImage);

//     // Create the SharpeningStep
//     SharpeningStep sharpeningStep;

//     // Apply sharpening adjustment
//     sharpeningStep.process(img);

//     // Get the sharpened image
//     cv::Mat sharpenedImage = img.getImage();

//     // Test sharpening on an edge (boundary of the circle)
//     int originalCirclePixelValue =
//         originalImage.at<cv::Vec3b>(50, 50)[0];  // Center of the circle
//     int sharpenedCirclePixelValue = sharpenedImage.at<cv::Vec3b>(50, 50)[0];

//     // Ensure the pixel inside the circle has changed after sharpening
//     ASSERT_NE(sharpenedCirclePixelValue, originalCirclePixelValue);

//     // Test sharpening on the edge of the circle
//     int originalEdgePixelValue =
//         originalImage.at<cv::Vec3b>(50, 80)[0];  // Edge of the circle
//     int sharpenedEdgePixelValue = sharpenedImage.at<cv::Vec3b>(50, 80)[0];

//     // Ensure the pixel at the edge of the circle has been sharpened
//     ASSERT_NE(sharpenedEdgePixelValue, originalEdgePixelValue);
// }

// Test case for no sharpening
TEST(SharpeningTest, NoSharpeningEffectIsAppliedCorrectly)
{
    // Create a simple image object (100x100 color gradient)
    cv::Mat originalImage(100, 100, CV_8UC3);
    for (int i = 0; i < 100; ++i)
    {
        for (int j = 0; j < 100; ++j)
        {
            originalImage.at<cv::Vec3b>(i, j) =
                cv::Vec3b(j, 0, 255 - j);  // Blue to Red gradient
        }
    }
    Image img(originalImage);

    // Create the SharpeningStep with a kernel that does nothing
    SharpeningStep
        sharpeningStep;  // Adjust if your sharpening step can accept parameters

    // Apply sharpening adjustment
    sharpeningStep.process(img);

    // Get the sharpened image
    cv::Mat sharpenedImage = img.getImage();

    // Check that the sharpened image is the same as the original image
    ASSERT_EQ(sharpenedImage.at<cv::Vec3b>(50, 50)[0],
              originalImage.at<cv::Vec3b>(50, 50)[0]);  // Blue channel
    ASSERT_EQ(sharpenedImage.at<cv::Vec3b>(50, 50)[1],
              originalImage.at<cv::Vec3b>(50, 50)[1]);  // Green channel
    ASSERT_EQ(sharpenedImage.at<cv::Vec3b>(50, 50)[2],
              originalImage.at<cv::Vec3b>(50, 50)[2]);  // Red channel
}
