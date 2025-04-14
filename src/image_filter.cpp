#include "image_filter.hpp"

#include "utilities.hpp"

ImageFilter::ImageFilter(int width, int height) : width_(width),
                                                  height_(height),
                                                  highPassFilter_(getHighPassFilter(height_, width_)),
                                                  apodizationWindow_(getApodizationWindow(width_, height_, std::min(width_, height_))) {
}

cv::Mat ImageFilter::GetFilteredImage(const cv::Mat& img) const {
    cv::Mat apodized = img.mul(apodizationWindow_);
    cv::Mat dftResult = fft(apodized);
    cv::Mat filtered = fftShift(dftResult);
    cv::multiply(filtered, highPassFilter_, filtered);

    std::vector<cv::Mat> channels(2);
    cv::split(filtered, channels);
    cv::Mat magnitude;
    cv::magnitude(channels[0], channels[1], magnitude);
    return magnitude;
}
