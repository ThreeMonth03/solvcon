/*
 * Copyright (c) 2022, solvcon team <contact@solvcon.net>
 * BSD 3-Clause License, see COPYING
 */

#include <solvcon/buffer/pymod/wrap_SimpleArray.hpp> // Must be the first include.

namespace solvcon
{

namespace python
{

void wrap_SimpleArray(pybind11::module & mod)
{
    namespace py = pybind11;

    py::class_<solvcon::detail::MatmulRoute>(mod, "MatmulRoute")
        .def_property_readonly("kernel", &solvcon::detail::MatmulRoute::kernel_name)
        .def_property_readonly("selected_by_auto", &solvcon::detail::MatmulRoute::selected_by_auto)
        .def_property_readonly("eager_pack_lhs", &solvcon::detail::MatmulRoute::eager_pack_lhs)
        .def_property_readonly("eager_pack_rhs", &solvcon::detail::MatmulRoute::eager_pack_rhs)
        .def_property_readonly("scratch_pack_lhs", &solvcon::detail::MatmulRoute::scratch_pack_lhs)
        .def_property_readonly("scratch_pack_rhs", &solvcon::detail::MatmulRoute::scratch_pack_rhs)
        .def(
            "__repr__",
            [](solvcon::detail::MatmulRoute const & route)
            { return std::string("<MatmulRoute kernel='") + route.kernel_name() + "'>"; });

    wrap_SimpleArray_bool(mod);
    wrap_SimpleArray_int(mod);
    wrap_SimpleArray_uint(mod);
    wrap_SimpleArray_float(mod);
    wrap_SimpleArray_complex(mod);
}

} /* end namespace python */

} /* end namespace solvcon */

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
