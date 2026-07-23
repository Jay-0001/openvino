#pragma once

#include "offline_correlation_types.hpp"

namespace ov::intel_gpu::gtpin {

std::vector<BuildImplementationRecord> parse_build_implementations_info(
    const std::filesystem::path& info_path);

}  // namespace ov::intel_gpu::gtpin
