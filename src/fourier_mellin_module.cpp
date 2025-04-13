#include "fourier_mellin.hpp"

#include <opencv2/opencv.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <iomanip>
#include <sstream>

#ifndef MODULE_NAME
#error "MODULE_NAME is not defined"
#endif

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

namespace py = pybind11;
using namespace pybind11::literals;

template<unsigned int Channels>
cv::Mat numpy_to_mat(const py::array_t<float>& input) {
    py::buffer_info buf = input.request();
    cv::Mat mat(buf.shape[0], buf.shape[1], CV_32FC(Channels), (float*)buf.ptr);
    return mat;
}

template<>
cv::Mat numpy_to_mat<0>(const py::array_t<float>& input){
    py::buffer_info buf = input.request();
    int type;
    int channels = buf.ndim == 3 ? buf.shape[2] : 1;

    switch(channels){
        case 1: type = CV_32FC1; break;
        case 2: type = CV_32FC2; break;
        case 3: type = CV_32FC3; break;
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
        data.data()
    );
}

struct PyLogPolarMap{
    int logPolarSize;
    double logBase;
    py::array_t<float> xMap;
    py::array_t<float> yMap;

    static PyLogPolarMap ConvertFromLogPolarMap(const LogPolarMap& polarMap){
        return PyLogPolarMap{
            .logPolarSize = polarMap.logPolarSize,
            .logBase = polarMap.logBase,
            .xMap = mat_to_numpy(polarMap.xMap),
            .yMap = mat_to_numpy(polarMap.yMap),
        };
    }

    LogPolarMap ConvertToLogPolarMap(){
        return LogPolarMap{
            .logPolarSize = logPolarSize,
            .logBase = logBase,
            .xMap = numpy_to_mat<1>(xMap),
            .yMap = numpy_to_mat<1>(yMap),
        };
    }
};

template <typename T>
std::string to_string_with_precision(const T value, const int n=2){
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << value;
    return std::move(out).str();
}

PYBIND11_MODULE(MODULE_NAME, m) {
    py::class_<Transform>(m, "Transform")
        .def(py::init<>())
        .def(py::init<double, double, double, double, double>())
        .def("__repr__", [](const Transform& t){
            return "<" TOSTRING(MODULE_NAME) ".Transform x_offset=" + to_string_with_precision(t.GetOffsetX(), 2) + ", y_offset=" + to_string_with_precision(t.GetOffsetY(), 2) + ", rotation=" + to_string_with_precision(t.GetRotation(), 2) + ", scale=" + to_string_with_precision(t.GetScale(), 2) + ", response=" + to_string_with_precision(t.GetResponse(), 2) + ">";
        })
        .def("__mul__",[](const Transform &a, const Transform& b) {
            return a * b;
        }, py::is_operator())
        .def("get_matrix",[](const Transform &a) {
            cv::Mat matrix = a.GetMatrix();
            return py::array_t<double>(
                {matrix.rows, matrix.cols},
                {matrix.step[0], matrix.step[1]},
                matrix.ptr<double>());
        }, "Get Matrix")
        .def("get_inverse_matrix",[](const Transform &a) {
            cv::Mat matrix = a.GetMatrixInverse();
            return py::array_t<double>(
                {matrix.rows, matrix.cols},
                {matrix.step[0], matrix.step[1]},
                matrix.ptr<double>());
        }, "Get Inverse Matrix")
        .def("get_inverse",[](const Transform &a) {
            return a.GetInverse();
        }, "Get Inverse Transform")

        .def("x", [](const Transform& t) {
            return t.GetOffsetX();
        }, "Get X Offset")
        .def("y", [](const Transform& t) {
            return t.GetOffsetY();
        }, "Get Y Offset")
        .def("scale", [](const Transform& t) {
            return t.GetScale();
        }, "Get Scale")
        .def("rotation", [](const Transform& t) {
            return t.GetRotation();
        }, "Get Rotation")
        .def("response", [](const Transform& t) {
            return t.GetResponse();
        }, "Get Response")
        .def("to_dict", [](const Transform& t){
            return py::dict("x"_a=t.GetOffsetX(), "y"_a=t.GetOffsetY(), "scale"_a=t.GetScale(), "rotation"_a=t.GetRotation(), "response"_a=t.GetResponse());
        });

    py::class_<FourierMellin>(m, "FourierMellin")
        .def(py::init<int, int>())
        .def("process_image", [](const FourierMellin& fm, py::array_t<float> img) -> auto {
            auto mat = numpy_to_mat<1>(img);
            auto matProcessed = fm.GetProcessImage(mat);
            return mat_to_numpy(matProcessed);
        }, "Process Image")
        .def("register_image", [](const FourierMellin& fm, const py::array_t<float>& img0, const py::array_t<float>& img1) -> auto {
            auto mat0 = numpy_to_mat<0>(img0);
            auto mat1 = numpy_to_mat<0>(img1);
            auto[transformed, transform] = fm.GetRegisteredImage(mat0, mat1);
            return std::make_tuple(mat_to_numpy(transformed), transform);
        }, "Register Image");

    m.def("process_image", [](const py::array_t<float>& img, const py::array_t<float>& highPassFilter, const py::array_t<float>& apodizationWindow, PyLogPolarMap logPolarMap){
        auto logPolarMap2 = logPolarMap.ConvertToLogPolarMap();
        auto img2 = numpy_to_mat<1>(img);
        auto highPassFilter2 = numpy_to_mat<2>(highPassFilter);
        auto apodizationWindow2 = numpy_to_mat<1>(apodizationWindow);
        auto logPolarImg = getProcessedImage(img2, highPassFilter2, apodizationWindow2, logPolarMap2);
        return mat_to_numpy(logPolarImg);
    }, "Process Image");

    m.def("register_image", [](const py::array_t<float>& img0, const py::array_t<float>& img1, const py::array_t<float>& logPolar0, const py::array_t<float>& logPolar1, PyLogPolarMap logPolarMap){
        auto mat0 = numpy_to_mat<1>(img0);
        auto mat1 = numpy_to_mat<1>(img1);
        auto matLogPolar0 = numpy_to_mat<1>(logPolar0);
        auto matLogPolar1 = numpy_to_mat<1>(logPolar1);
        auto logPolarMap2 = logPolarMap.ConvertToLogPolarMap();
        return registerGrayImage(mat0, mat1, matLogPolar0, matLogPolar1, logPolarMap2);
    }, "Register Images");

    m.def("get_transformed", [](const py::array_t<float>& img, Transform transform){
        auto mat = numpy_to_mat<0>(img);
        auto transformed = getTransformed(mat, transform);
        return mat_to_numpy(transformed);
    }, "Process Image");
}
