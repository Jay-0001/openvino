#pragma once

#include "offline_correlation_types.hpp"

namespace ov::intel_gpu::gtpin {

std::vector<GTPinKernelOccurrence> parse_gtpin_profile(
    const std::filesystem::path& profile_path);

}  // namespace ov::intel_gpu::gtpin
