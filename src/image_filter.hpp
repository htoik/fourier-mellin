#ifndef __IMAGE_FILTER_H__
#define __IMAGE_FILTER_H__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

class ImageFilter {
   public:
    ImageFilter(int width, int height);

    cv::Mat GetFilteredImage(const cv::Mat& img) const;

    ~ImageFilter() = default;
    ImageFilter(const ImageFilter&) = delete;
    ImageFilter(ImageFilter&&) = delete;
    ImageFilter& operator=(const ImageFilter&) = delete;
    ImageFilter& operator=(ImageFilter&&) = delete;

   private:
    int width_, height_;
    cv::Mat highPassFilter_;
    cv::Mat apodizationWindow_;
};

#endif  // __IMAGE_FILTER_H__