#include "log_polar_map.hpp"

#include <numbers>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

constexpr long double pi = std::numbers::pi_v<long double>;

LogPolarMap::LogPolarMap() : LogPolarMap(0, 0) {
}

LogPolarMap::LogPolarMap(int width, int height) : width_(width),
                                                  height_(height),
                                                  logPolarSize_(0),
                                                  logBase_(0.0) {
    ConstructMaps();
}

cv::Mat LogPolarMap::ConvertToLogPolar(const cv::Mat& img) const {
    cv::Mat log_polar;
    cv::remap(img, log_polar, xMap_, yMap_, cv::INTER_LANCZOS4, cv::BORDER_CONSTANT, cv::Scalar());
    return log_polar;
}

void LogPolarMap::ConstructMaps() {
    logPolarSize_ = std::max(width_, height_);
    logBase_ = std::exp(std::log(logPolarSize_ * 1.5 / 2.0) / logPolarSize_);

    float ellipse_coefficient = height_ / (float)width_;

    xMap_ = cv::Mat(logPolarSize_, logPolarSize_, CV_32FC1);
    yMap_ = cv::Mat(logPolarSize_, logPolarSize_, CV_32FC1);

    for (int i = 0; i < logPolarSize_; i++) {
        float angle = -(pi / logPolarSize_) * i;
        float cos_angle = std::cos(angle) / ellipse_coefficient;
        float sin_angle = std::sin(angle);

        for (int j = 0; j < logPolarSize_; j++) {
            float scale = std::pow(logBase_, j);
            xMap_.at<float>(i, j) = scale * cos_angle + width_ / 2.0f;
            yMap_.at<float>(i, j) = scale * sin_angle + height_ / 2.0f;
        }
    }
}

int LogPolarMap::GetLogPolarSize() const {
    return logPolarSize_;
}

double LogPolarMap::GetLogBase() const {
    return logBase_;
}
