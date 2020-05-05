#pragma once
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


class DahuApplication;

class PyDahuApplication
{

public:
  PyDahuApplication(pybind11::array input);
  ~PyDahuApplication();

  // Set the markers as an array of 2D coordinates
  void setMarkers(pybind11::array_t<int, pybind11::array::c_style | pybind11::array::forcecast> fg_points,
                  pybind11::array_t<int, pybind11::array::c_style | pybind11::array::forcecast> bg_points);

  // Return an image (n x m) that holds the distance to the background
  pybind11::array getBackgroundDistanceImage() const;
  pybind11::array getForegroundDistanceImage() const;

private:
  std::unique_ptr<DahuApplication> m_app;
};
