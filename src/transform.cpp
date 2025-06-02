#include "transform.hpp"

#include <iomanip>
#include <numbers>

constexpr long double pi = std::numbers::pi_v<long double>;

cv::Mat Transform::GetMatrix() const {
  cv::Mat transform = cv::Mat::zeros(3, 3, CV_64F);

  double c = std::cos(rotation * (pi / 180.0));
  double s = std::sin(rotation * (pi / 180.0));

  transform.at<double>(0, 0) = scale * c;
  transform.at<double>(0, 1) = -scale * s;
  transform.at<double>(0, 2) = x;
  transform.at<double>(1, 0) = scale * s;
  transform.at<double>(1, 1) = scale * c;
  transform.at<double>(1, 2) = y;
  transform.at<double>(2, 0) = 0.0;
  transform.at<double>(2, 1) = 0.0;
  transform.at<double>(2, 2) = 1.0;

  return transform;
}

cv::Mat Transform::GetMatrixInverse() const { return GetMatrix().inv(); }

Transform Transform::operator*(const Transform &rhs) const {
  // TODO: How should responses be combined?
  // double response = (response_ + rhs.response_) * 0.5;
  cv::Mat combined = GetMatrix() * rhs.GetMatrix();

  double x = combined.at<double>(0, 2);
  double y = combined.at<double>(1, 2);
  double scale =
      std::hypot(combined.at<double>(0, 0), combined.at<double>(0, 1));
  double rotation =
      std::atan2(combined.at<double>(1, 0), combined.at<double>(0, 0)) *
      (180.0 / pi);
  double response2 = std::min(response, rhs.response);

  return Transform{x, y, scale, rotation, response2};
}

std::ostream &operator<<(std::ostream &os, const Transform &t) {
  os << std::fixed << std::setprecision(2) << "Transform(" << t.x << ", " << t.y
     << ", " << t.scale << ", " << t.rotation << ", " << t.response << ")";
  return os;
}

Transform &Transform::operator*=(const Transform &rhs) {
  *this = *this * rhs;
  return *this;
}

Transform Transform::GetInverse() const {
  Transform inv;

  inv.scale = 1.0 / scale;
  inv.rotation = -rotation;

  double theta = rotation * (pi / 180.0);
  double cos_theta = std::cos(theta);
  double sin_theta = std::sin(theta);

  inv.x = -(cos_theta * x + sin_theta * y) / scale;
  inv.y = (sin_theta * x - cos_theta * y) / scale;

  inv.response = response;
  return inv;
}
