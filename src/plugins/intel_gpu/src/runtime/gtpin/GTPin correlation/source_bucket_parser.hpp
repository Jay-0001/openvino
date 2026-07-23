#pragma once

#include "offline_correlation_types.hpp"

namespace ov::intel_gpu::gtpin {

std::vector<SourceBucketRecord> parse_source_bucket_files(
    const std::vector<std::filesystem::path>& source_bucket_paths);

}  // namespace ov::intel_gpu::gtpin
