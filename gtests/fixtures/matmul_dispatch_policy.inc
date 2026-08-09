static_assert(MATMUL_POLICY_SCHEMA_VERSION == 1);

template <typename Eligible>
inline std::optional<MatmulKernel> select_calibrated_gemm(
    MatmulFacts const & facts,
    Eligible && eligible_kernel)
{
    if (facts.operation != MatmulOperation::Gemm)
    {
        return std::nullopt;
    }

    bool const dtype_supported =
        facts.dtype == MatmulDataType::Float64;
    if (!dtype_supported)
    {
        return std::nullopt;
    }

    bool const backend_supported =
        facts.backend == MatmulBackend::Accelerate ||
        facts.backend == MatmulBackend::Cblas ||
        facts.backend == MatmulBackend::Mkl ||
        facts.backend == MatmulBackend::None;
    if (!backend_supported)
    {
        return std::nullopt;
    }

    bool const lhs_layout_supported =
        facts.lhs_layout == MatmulLayout::RowMajor;
    if (!lhs_layout_supported)
    {
        return std::nullopt;
    }

    bool const rhs_layout_supported =
        facts.rhs_layout == MatmulLayout::RowMajor;
    if (!rhs_layout_supported)
    {
        return std::nullopt;
    }

    if (facts.rows < 3 || facts.rows > 3)
    {
        return std::nullopt;
    }
    if (facts.columns < 7 || facts.columns > 7)
    {
        return std::nullopt;
    }
    if (facts.inner_size < 4 || facts.inner_size > 4)
    {
        return std::nullopt;
    }
    if (facts.batch_size < 1 || facts.batch_size > 1)
    {
        return std::nullopt;
    }
    if (facts.has_batch_axes)
    {
        return std::nullopt;
    }
    if (facts.lhs_reused)
    {
        return std::nullopt;
    }
    if (facts.rhs_reused)
    {
        return std::nullopt;
    }
    if (facts.lhs_zero_batch_stride)
    {
        return std::nullopt;
    }
    if (facts.rhs_zero_batch_stride)
    {
        return std::nullopt;
    }

    auto const eligible =
        [&facts, &eligible_kernel](MatmulKernel kernel)
    {
        bool const calibrated_loop =
            kernel == MatmulKernel::GenericIjk ||
            kernel == MatmulKernel::DynamicIkj;
        std::size_t const contraction_work =
            static_cast<std::size_t>(facts.rows) *
            static_cast<std::size_t>(facts.columns) *
            static_cast<std::size_t>(facts.inner_size);
        if (calibrated_loop && contraction_work >
            84)
        {
            return false;
        }
        return eligible_kernel(kernel);
    };

    if (eligible(MatmulKernel::DynamicIkj))
    {
        return MatmulKernel::DynamicIkj;
    }
    return std::nullopt;
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
