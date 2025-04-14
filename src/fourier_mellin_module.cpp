#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iomanip>
#include <opencv2/opencv.hpp>
#include <sstream>

#include "fourier_mellin.hpp"

#ifndef MODULE_NAME
#error "MODULE_NAME is not defined"
#endif

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

namespace py = pybind11;
using namespace pybind11::literals;

template <unsigned int Channels>
cv::Mat numpy_to_mat(const py::array_t<float>& input) {
    py::buffer_info buf = input.request();
    cv::Mat mat(buf.shape[0], buf.shape[1], CV_32FC(Channels), (float*)buf.ptr);
    return mat;
}

template <>
cv::Mat numpy_to_mat<0>(const py::array_t<float>& input) {
    py::buffer_info buf = input.request();
    int type;
    int channels = buf.ndim == 3 ? buf.shape[2] : 1;

    switch (channels) {
        case 1:
            type = CV_32FC1;
            break;
        case 2:
            type = CV_32FC2;
            break;
        case 3:
            type = CV_32FC3;
            break;
        // case 4: type = CV_32FC4; break;
        default:
            throw std::runtime_error("Invalid channel count: " + std::to_string(channels));
            break;
    }
    cv::Mat mat(buf.shape[0], buf.shape[1], CV_32FC(channels), (float*)buf.ptr);
    return mat;
}

py::array_t<float> mat_to_numpy(const cv::Mat& mat) {
    py::ssize_t rows = mat.rows;
    py::ssize_t cols = mat.cols;
    py::ssize_t channels = mat.channels();
    py::ssize_t total_size = rows * cols * channels;

    std::vector<float> data(total_size);
    if (mat.isContinuous()) {
        std::memcpy(data.data(), mat.ptr<float>(), total_size * sizeof(float));
    } else {
        for (int i = 0; i < rows; ++i) {
            std::memcpy(data.data() + i * cols * channels, mat.ptr<float>(i), cols * channels * sizeof(float));
        }
    }

    return py::array_t<float>(
        {rows, cols, channels},
        {cols * channels * sizeof(float), channels * sizeof(float), sizeof(float)},
        data.data());
}

template <typename T>
std::string to_string_with_precision(const T value, const int n = 2) {
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << value;
    return std::move(out).str();
}

PYBIND11_MODULE(MODULE_NAME, m) {
    py::class_<Transform>(m, "Transform")
        .def(py::init<>())
        .def("__repr__", [](const Transform& t) {
            // TODO: use the operator<< format previously defined instead for consistency
            return "<" TOSTRING(MODULE_NAME) ".Transform x_offset=" + to_string_with_precision(t.x, 2) + ", y_offset=" + to_string_with_precision(t.y, 2) + ", rotation=" + to_string_with_precision(t.rotation, 2) + ", scale=" + to_string_with_precision(t.scale, 2) + ", response=" + to_string_with_precision(t.response, 2) + ">";
        })
        .def_readwrite("x", &Transform::x, "x offset")
        .def_readwrite("y", &Transform::y, "y offset")
        .def_readwrite("scale", &Transform::scale, "scale")
        .def_readwrite("rotation", &Transform::rotation, "rotation")
        .def_readwrite("response", &Transform::response, "response")
        .def("__mul__", [](const Transform& a, const Transform& b) { return a * b; }, py::is_operator())
        .def("get_matrix", [](const Transform& a) {
            cv::Mat matrix = a.GetMatrix();
            return py::array_t<double>(
                {matrix.rows, matrix.cols},
                {matrix.step[0], matrix.step[1]},
                matrix.ptr<double>()); }, "Get Matrix")
        .def("get_inverse_matrix", [](const Transform& a) {
            cv::Mat matrix = a.GetMatrixInverse();
            return py::array_t<double>(
                {matrix.rows, matrix.cols},
                {matrix.step[0], matrix.step[1]},
                matrix.ptr<double>()); }, "Get Inverse Matrix")
        .def("get_inverse", [](const Transform& a) { return a.GetInverse(); }, "Get Inverse Transform")
        .def("to_dict", [](const Transform& t) { return py::dict("x"_a = t.x, "y"_a = t.y, "scale"_a = t.scale, "rotation"_a = t.rotation, "response"_a = t.response); });

    py::class_<FourierMellin>(m, "FourierMellin")
        .def(py::init<std::string_view>())
        .def(py::init([](py::array_t<float> reference) {
            auto referenceMat = numpy_to_mat<0>(reference).clone();
            return new FourierMellin(referenceMat);
        }))
        .def("register_image", [](const FourierMellin& fm, std::string_view target_fp) -> auto {
                    auto[transformed, transform] = fm.GetRegisteredImage(target_fp);
                    return std::make_tuple(mat_to_numpy(transformed), transform); }, "Register target image to reference and return aligned target")
        .def("register_image", [](const FourierMellin& fm, py::array_t<float> target) -> auto {
            auto targetMat = numpy_to_mat<0>(target).clone();
            auto[transformed, transform] = fm.GetRegisteredImage(targetMat);
            return std::make_tuple(mat_to_numpy(transformed), transform); }, "Register target image to reference and return aligned target");

    m.def("get_transformed", [](const py::array_t<float>& img, Transform transform) {
        auto mat = numpy_to_mat<0>(img);
        auto transformed = getTransformed(mat, transform);
        return mat_to_numpy(transformed); }, "Process Image");
}
