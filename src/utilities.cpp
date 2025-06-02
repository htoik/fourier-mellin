#include "utilities.hpp"

#include <iostream>
#include <numbers>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

constexpr long double pi = std::numbers::pi_v<long double>;

cv::Mat fft(const cv::Mat &img) {
  cv::Mat planes[] = {cv::Mat_<float>(img), cv::Mat::zeros(img.size(), CV_32F)};
  cv::Mat complex;
  cv::merge(planes, 2, complex);
  cv::dft(complex, complex, cv::DFT_COMPLEX_OUTPUT);
  return complex;
}

cv::Mat fftShift(const cv::Mat &in) {
  cv::Mat out = in.clone();
  int cx = in.cols / 2;
  int cy = in.rows / 2;

  int cx1 = (in.cols % 2 == 0) ? cx : cx + 1;
  int cy1 = (in.rows % 2 == 0) ? cy : cy + 1;

  cv::Mat q0(out, cv::Rect(0, 0, cx, cy));
  cv::Mat q1(out, cv::Rect(cx, 0, cx1, cy));
  cv::Mat q2(out, cv::Rect(0, cy, cx, cy1));
  cv::Mat q3(out, cv::Rect(cx, cy, cx1, cy1));

  cv::Mat tmp;
  q0.copyTo(tmp);
  q3.copyTo(q0);
  tmp.copyTo(q3);

  q1.copyTo(tmp);
  q2.copyTo(q1);
  tmp.copyTo(q2);
  return out;
}

cv::Mat linspace(float min, float max, size_t count) {
  cv::Mat result = cv::Mat::zeros(count, 1, CV_32F);
  float step = (max - min) / (count - 1);
  for (size_t i = 0; i < count; i++) {
    result.at<float>(i, 0) = min + step * i;
  }
  return result;
}

cv::Mat getHighPassFilter(int rows, int cols) {
  cv::Mat y = linspace(-pi / 2.0, pi / 2.0, rows);
  cv::Mat x = linspace(-pi / 2.0, pi / 2.0, cols).t();
  cv::Mat yMat = cv::repeat(y, 1, cols);
  cv::Mat xMat = cv::repeat(x, rows, 1);

  cv::Mat temp = yMat.mul(yMat) + xMat.mul(xMat);
  cv::sqrt(temp, temp);

  cv::Mat filter = cv::Mat(temp.size(), temp.type());
  for (int i = 0; i < temp.rows; i++) {
    for (int j = 0; j < temp.cols; j++) {
      filter.at<float>(i, j) = std::cos(temp.at<float>(i, j));
    }
  }

  filter = filter.mul(filter);
  filter = -filter + 1.0;

  cv::Mat channels[] = {filter, filter};
  cv::Mat filterTwoChannel;
  cv::merge(channels, 2, filterTwoChannel);

  cv::Mat filterConverted;
  filterTwoChannel.convertTo(filterConverted, CV_32F);
  return filterConverted;
}

cv::Mat getApodizationWindow(int cols, int rows, int radius) {
  cv::Mat hanningWindow;
  cv::createHanningWindow(hanningWindow, cv::Size(radius, radius), CV_32F);
  cv::resize(hanningWindow, hanningWindow, cv::Size(cols, rows), 0.0, 0.0,
             cv::InterpolationFlags::INTER_CUBIC);
  return hanningWindow;
}

cv::Mat getTransformed(const cv::Mat &img, const Transform &transform) {
  // TODO: Interpolation

  cv::Point2f center(img.cols / 2.f, img.rows / 2.f);

  cv::Mat rotationMatrix =
      cv::getRotationMatrix2D(center, transform.rotation, transform.scale);
  rotationMatrix.at<double>(0, 2) += transform.x;
  rotationMatrix.at<double>(1, 2) += -transform.y;

  cv::Mat transformed = img.clone();

  cv::warpAffine(transformed, transformed, rotationMatrix, transformed.size(),
                 cv::INTER_CUBIC);
  return transformed;
}

cv::Mat getCropped(const cv::Mat &img, double x1, double y1, double x2,
                   double y2) {
  int x_start = static_cast<int>(x1);
  int y_start = static_cast<int>(y1);
  int width = static_cast<int>(x2 - x1);
  int height = static_cast<int>(y2 - y1);

  cv::Rect roi(x_start, y_start, width, height);

  cv::Mat cropped = img(roi);
  return cropped;
}
