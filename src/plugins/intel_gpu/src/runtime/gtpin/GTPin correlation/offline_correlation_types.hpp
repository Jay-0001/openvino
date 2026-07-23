#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace ov::intel_gpu::gtpin {

enum class KernelCategory {
    main_compute,
    support_reorder,
    support_weight_reorder,
    auxiliary,
    output,
    unknown,
};

enum class PrimitiveMappingKind {
    unique_primitive,
    shared_primitives,
    unmatched,
};

struct GTPinKernelOccurrence {
    std::string name;
    std::string hash_id;
    std::string type;
    std::string platform;
    std::string execution_descriptor;
    std::optional<uint32_t> simd;
    std::optional<uint64_t> frequency;
    std::optional<uint64_t> total_cycles;
    std::optional<uint64_t> average_cycles;
    std::optional<uint64_t> skipped;
    size_t file_line = 0;
    size_t occurrence_index = 0;
};

struct BuildImplementationRecord {
    std::string primitive_id;
    std::string primitive_type;
    std::string implementation;
    std::string kernel_entry;
    std::string batch_hash;
    std::string source_info_file;
    bool constant = false;
    bool output = false;
    bool in_data_flow = false;
    std::optional<uint32_t> in_shape_of_subgraph;
};

struct SourceBucketRecord {
    std::filesystem::path source_path;
    std::string batch_hash;
    std::optional<uint32_t> program_id;
    std::optional<uint32_t> bucket_id;
    std::optional<uint32_t> part_id;
    std::vector<std::string> kernel_entries;
};

struct CorrelationOccurrenceRow {
    GTPinKernelOccurrence occurrence;
    std::optional<BuildImplementationRecord> build_record;
    std::vector<std::filesystem::path> source_paths;
    KernelCategory category = KernelCategory::unknown;
    PrimitiveMappingKind mapping_kind = PrimitiveMappingKind::unmatched;
    size_t primitive_fanout = 0;
    std::vector<std::string> mapped_primitive_ids;
};

struct KernelIdentitySummary {
    std::string kernel_entry;
    std::string primitive_id;
    std::string primitive_type;
    std::string implementation;
    std::string batch_hash;
    std::vector<std::filesystem::path> source_paths;
    KernelCategory category = KernelCategory::unknown;
    size_t occurrence_count = 0;
    std::vector<std::string> gtpin_names;
    std::vector<std::string> hash_ids;
    std::vector<std::string> execution_descriptors;
    PrimitiveMappingKind mapping_kind = PrimitiveMappingKind::unmatched;
    size_t primitive_fanout = 0;
    std::vector<std::string> mapped_primitive_ids;
};

struct OfflineCorrelationOptions {
    std::filesystem::path gtpin_profile_path;
    std::vector<std::filesystem::path> build_info_paths;
    std::vector<std::filesystem::path> source_bucket_paths;
    std::optional<std::filesystem::path> kernel_occurrences_csv_path;
    std::optional<std::filesystem::path> kernel_identity_summary_csv_path;
};

struct OfflineCorrelationArtifacts {
    std::vector<GTPinKernelOccurrence> gtpin_occurrences;
    std::vector<BuildImplementationRecord> build_records;
    std::vector<SourceBucketRecord> source_buckets;
    std::vector<CorrelationOccurrenceRow> occurrence_rows;
    std::vector<KernelIdentitySummary> identity_summaries;
};

std::string to_string(KernelCategory category);
std::string to_string(PrimitiveMappingKind mapping_kind);

}  // namespace ov::intel_gpu::gtpin
