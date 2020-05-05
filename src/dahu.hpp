#pragma once

#include <mln/core/image/experimental/ndimage_fwd.hpp>
#include <mln/core/experimental/point.hpp>
#include <array>
#include <memory>

enum dahu_label_t
{
  DAHU_BACKGROUND = 0,
  DAHU_FOREGROUND = 1,
};


class DahuApplication
{
public:
  // Create a ToS and initialize the application
  DahuApplication(const mln::experimental::image2d<uint8_t>& input);
  ~DahuApplication();

  // Set/update the markers
  void setMarkers(const mln::experimental::point2d *fg_points, std::size_t fg_size,
                  const mln::experimental::point2d *bg_points, std::size_t bg_size);

  using distance_result_t = std::array<uint8_t, 2>;
  // Getters for the distance images
  mln::ndbuffer_image getForegroundDistanceImage() const;
  mln::ndbuffer_image getBackgroundDistanceImage() const;

private:
  struct impl_t;

  std::unique_ptr<impl_t> m_impl;
};




