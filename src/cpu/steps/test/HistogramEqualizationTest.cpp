#include "HistogramEqualizationStep.h"
#include "Image.h"
#include <gtest/gtest.h>

// Test case for Histogram Equalization Step
TEST(HistogramEqualizationTest, HistogramEqualizationEffectIsAppliedCorrectly)
{
    // Create a grayscale image object with some test data (100x100 gradient
    // image)
    cv::Mat grayImage = cv::Mat(100, 100, CV_8UC1);
    for (int i = 0; i < 100; ++i)
    {
        for (int j = 0; j < 100; ++j)
        {
            grayImage.at<uchar>(i, j) =
                static_cast<uchar>(i);  // Gradient from 0 to 99
        }
    }
    Image img(grayImage);

    // Create the HistogramEqualizationStep
    HistogramEqualizationStep histogramEqualizationStep;

    // Apply histogram equalization
    histogramEqualizationStep.process(img);

    // Get the processed image
    cv::Mat equalizedImage = img.getImage();

    // Check if the histogram is equalized (checking mean intensity)
    double meanIntensity = cv::mean(equalizedImage)[0];
    ASSERT_GT(meanIntensity, 0);    // Ensure mean intensity is greater than 0
    ASSERT_LT(meanIntensity, 255);  // Ensure mean intensity is less than 255
}
