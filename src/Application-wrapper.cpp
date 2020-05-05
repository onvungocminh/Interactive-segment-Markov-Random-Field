#include "Application-wrapper.hpp"
#include "dahu.hpp"


#include <pybind11/pybind11.h>
#include <mln/core/image/experimental/ndimage.hpp>
#include <exception>

#include "ndimage_buffer_helper.hpp"

namespace py = pybind11;




PyDahuApplication::PyDahuApplication(py::array input)
{
  auto img = mln::py::ndimage_from_buffer(input);

  py::gil_scoped_release release;

  auto casted = img.cast_to<uint8_t, 2>();
  if (!casted)
    throw std::runtime_error("Invalid input image (Must be 2D-uint8).");

  m_app = std::make_unique<DahuApplication>(*casted);
}

PyDahuApplication::~PyDahuApplication() {}

void PyDahuApplication::setMarkers(py::array_t<int, py::array::c_style | py::array::forcecast> fg_points,
                                   py::array_t<int, py::array::c_style | py::array::forcecast> bg_points)
{
  if (fg_points.ndim() != 2 || fg_points.shape(1) != 2)
    throw std::runtime_error("Foreground image has an invalid shape (Must be Nx2).");
  if (bg_points.ndim() != 2 || bg_points.shape(1) != 2)
    throw std::runtime_error("Background image has an invalid shape (Must be Nx2).");

  const mln::experimental::point2d* fg_buf  = (const mln::experimental::point2d*)fg_points.data();
  const mln::experimental::point2d* bg_buf  = (const mln::experimental::point2d*)bg_points.data();
  std::size_t                       fg_size = fg_points.shape(0);
  std::size_t                       bg_size = bg_points.shape(0);

  m_app->setMarkers(fg_buf, fg_size, bg_buf, bg_size);
}


py::array PyDahuApplication::getForegroundDistanceImage() const
{
  py::array arr = mln::py::ndimage_to_buffer(m_app->getForegroundDistanceImage());
  return arr;
}

py::array PyDahuApplication::getBackgroundDistanceImage() const
{
  py::array arr = mln::py::ndimage_to_buffer(m_app->getBackgroundDistanceImage());
  return arr;
}



PYBIND11_MODULE(dahu, m)
{
  py::class_<PyDahuApplication>(m, "DahuApplication")
    .def(py::init<py::array>())
    .def_property_readonly("fg", &PyDahuApplication::getForegroundDistanceImage, ::py::return_value_policy::reference_internal)
    .def_property_readonly("bg", &PyDahuApplication::getBackgroundDistanceImage, ::py::return_value_policy::reference_internal)
    .def("setMarkers", &PyDahuApplication::setMarkers)
    ;
}
