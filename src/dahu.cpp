#include "dahu.hpp"
#include <mln/core/image/experimental/ndimage.hpp>
#include <mln/morpho/experimental/tos.hpp>
#include <fmt/core.h>


namespace
{
  struct barrier_attribute_t;
}

struct DahuApplication::impl_t
{
  using V = uint8_t;
  mln::experimental::image2d<V>                m_input;
  mln::morpho::experimental::component_tree<V> m_tree;
  mln::experimental::image2d<int>              m_nodemap;


  void _compute_dahu_distance_transform(const mln::experimental::point2d* fg_points, std::size_t fg_size,
                                        const mln::experimental::point2d* bg_points, std::size_t bg_size);

  // Cache data
  std::vector<barrier_attribute_t>    m_attributes;
  mln::experimental::image2d<uint8_t> m_fg_distance;
  mln::experimental::image2d<uint8_t> m_bg_distance;
};


DahuApplication::DahuApplication(const mln::experimental::image2d<uint8_t>& input)
{
  m_impl          = std::make_unique<impl_t>();
  m_impl->m_input = input;

  mln::experimental::image2d<int> nodemap;
  std::tie(m_impl->m_tree, nodemap) = mln::morpho::experimental::tos(input, {0,0});


  // Copy domain from Khalimsly to original domain
  {
    m_impl->m_nodemap.resize(input.domain());
    mln_foreach_new (auto px, m_impl->m_nodemap.new_pixels())
      px.val() = nodemap(2 * px.point());
  }

  std::size_t n = m_impl->m_tree.parent.size();
  m_impl->m_attributes.resize(n);
  m_impl->m_fg_distance.resize(input.domain());
  m_impl->m_bg_distance.resize(input.domain());
}


DahuApplication::~DahuApplication() {}



namespace
{
  struct barrier_attribute_t
  {
    // Take another atribute
    void take(const barrier_attribute_t& other)
    {
      this->vmin = std::min(this->vmin, other.vmin);
      this->vmax = std::max(this->vmax, other.vmax);
    }

    void take(uint8_t value)
    {
      this->vmin = std::min(this->vmin, value);
      this->vmax = std::max(this->vmax, value);
    }

    void init(uint8_t value)
    {
      this->vmin = value;
      this->vmax = value;
    }

    void init()
    {
      this->vmin = UINT8_MAX;
      this->vmax = 0;
    }


    // Return the barrier value for a given label
    uint8_t barrier_value() const { return vmax - vmin; }


    uint8_t vmin = UINT8_MAX;
    uint8_t vmax = 0;
    uint8_t best_barrier_value[2] = {UINT8_MAX, UINT8_MAX};
  };

  void compute_and_update_distances(const int*           parent,         //
                                    const uint8_t*       values,         //
                                    barrier_attribute_t* attr,           //
                                    std::size_t          n,              //
                                    int                  source_node_id, //
                                    int                  source_node_label)
  {
    // Reinitialize barrier values
    std::for_each(attr, attr + n, [](auto& x) { x.init(); });

    // Propagate upward
    {
      attr[source_node_id].init(values[source_node_id]);
      for (int x = source_node_id, q = parent[x]; q >= 0;)
      {
        attr[q].init(values[q]);
        attr[q].take(attr[x]);
        x = q; q = parent[x];
      }
    }

    // Propagate downward
    for (std::size_t x = 0; x < n; ++x)
    {
      if (attr[x].vmin > attr[x].vmax) // Not seen
      {
        attr[x].init(values[x]);
        attr[x].take(attr[parent[x]]);
      }

      if (attr[x].barrier_value() < attr[x].best_barrier_value[source_node_label])
        attr[x].best_barrier_value[source_node_label] = attr[x].barrier_value();
    }
  }

} // namespace

void DahuApplication::impl_t::_compute_dahu_distance_transform(const mln::experimental::point2d* fg_points,
                                                               std::size_t                       fg_size,
                                                               const mln::experimental::point2d* bg_points,
                                                               std::size_t                       bg_size)
{
  auto& par      = m_tree.parent;


  std::size_t nnodes = par.size();


  // Reinitialize data
  std::fill(m_attributes.begin(), m_attributes.end(), barrier_attribute_t{});

  // Labelize each node from markers
  for (std::size_t i = 0; i < bg_size; ++i)
  {
    //fmt::print("({},{},0)\n", bg_points[i].x(), bg_points[i].y());
    int q = m_nodemap(bg_points[i]);
    compute_and_update_distances(par.data(), m_tree.values.data(), m_attributes.data(), nnodes, q, DAHU_BACKGROUND);
  }
  for (std::size_t i = 0; i < fg_size; ++i)
  {
    //fmt::print("({},{},1)\n", fg_points[i].x(), fg_points[i].y());
    int q = m_nodemap(fg_points[i]);
    compute_and_update_distances(par.data(), m_tree.values.data(), m_attributes.data(), nnodes, q, DAHU_FOREGROUND);
  }

  // Cache result
  {
    mln_foreach_new(auto px, m_nodemap.new_pixels())
    {
      m_fg_distance(px.point()) = m_attributes[px.val()].best_barrier_value[1];
      m_bg_distance(px.point()) = m_attributes[px.val()].best_barrier_value[0];
    }
  }
}

void DahuApplication::setMarkers(const mln::experimental::point2d* fg_points, std::size_t fg_size,
                                 const mln::experimental::point2d* bg_points, std::size_t bg_size)
{
  m_impl->_compute_dahu_distance_transform(fg_points, fg_size, bg_points, bg_size);
}


mln::ndbuffer_image DahuApplication::getBackgroundDistanceImage() const
{
  return m_impl->m_bg_distance;
}

mln::ndbuffer_image DahuApplication::getForegroundDistanceImage() const
{
  return m_impl->m_fg_distance;
}

