#ifndef __UTILITIES_H__
#define __UTILITIES_H__

#include <opencv2/core/mat.hpp>

#include "transform.hpp"

cv::Mat fft(const cv::Mat& img);

cv::Mat fftShift(const cv::Mat& in);

cv::Mat linspace(float min, float max, size_t count);

cv::Mat getHighPassFilter(int rows, int cols);

cv::Mat getApodizationWindow(int cols, int rows, int radius);

cv::Mat getTransformed(const cv::Mat& img, const Transform& transform);

cv::Mat getCropped(const cv::Mat& img, double x1, double y1, double x2, double y2);

#endif  // __UTILITIES_H__