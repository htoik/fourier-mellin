#ifndef __UTILITIES_H__
#define __UTILITIES_H__

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "transform.hpp"

struct LogPolarMap {
    int logPolarSize;
    double logBase;
    cv::Mat xMap;
    cv::Mat yMap;
};

LogPolarMap createLogPolarMap(int cols, int rows);

cv::Mat getLogPolarImage(const cv::Mat& img, const cv::Mat& polarMapX, const cv::Mat& polarMapY);

cv::Mat fft(const cv::Mat& img);

cv::Mat fftShift(const cv::Mat& in);

cv::Mat getHighPassFilter(int rows, int cols);

cv::Mat getApodizationWindow(int cols, int rows, int radius);

cv::Mat getFilteredImage(const cv::Mat& gray, const cv::Mat& apodizationWindow, const cv::Mat& highPassFilter);

cv::Mat getTransformed(const cv::Mat& img, const Transform& transform);

cv::Mat getCropped(const cv::Mat& img, double x1, double y1, double x2, double y2);

cv::Mat getProcessedImage(const cv::Mat& img, const cv::Mat& highPassFilter, const cv::Mat& apodizationWindow, const LogPolarMap& logPolarMap);

Transform registerGrayImage(const cv::Mat& img0, const cv::Mat& img1, const cv::Mat& logPolar0, const cv::Mat& logPolar1, const LogPolarMap& logPolarMap);

// cv::Mat phaseCorrelateWithImage();

#endif  // __UTILITIES_H__