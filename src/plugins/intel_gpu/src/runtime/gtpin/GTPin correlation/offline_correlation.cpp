#include "offline_correlation.hpp"

#include "build_implementations_parser.hpp"
#include "gtpin_profile_parser.hpp"
#include "source_bucket_parser.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace ov::intel_gpu::gtpin {
namespace {

std::string trim(std::string value) {
    auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
    value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
    return value;
}

std::optional<uint64_t> parse_optional_u64(const std::string& raw) {
    const auto value = trim(raw);
    if (value.empty() || value == "NA" || value == "null") {
        return std::nullopt;
    }

    size_t parsed_chars = 0;
    const auto parsed_value = std::stoull(value, &parsed_chars);
    if (parsed_chars != value.size()) {
        return std::nullopt;
    }

    return parsed_value;
}

std::optional<uint32_t> parse_optional_u32(const std::string& raw) {
    const auto parsed_value = parse_optional_u64(raw);
    if (!parsed_value.has_value()) {
        return std::nullopt;
    }

    return static_cast<uint32_t>(*parsed_value);
}

bool parse_bool_value(const std::string& raw) {
    return trim(raw) == "true";
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open file: " + path.string());
    }

    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

std::vector<std::string> split_blocks_by_top_level_braces(const std::string& content) {
    std::vector<std::string> blocks;
    std::string current_block;
    int brace_depth = 0;

    for (char ch : content) {
        if (ch == '{') {
            if (brace_depth == 0) {
                current_block.clear();
            }
            ++brace_depth;
        }

        if (brace_depth > 0) {
            current_block.push_back(ch);
        }

        if (ch == '}') {
            --brace_depth;
            if (brace_depth == 0 && !current_block.empty()) {
                blocks.push_back(current_block);
                current_block.clear();
            }
        }
    }

    return blocks;
}

std::string extract_first_match(const std::string& text, const std::regex& pattern) {
    std::smatch match;
    if (!std::regex_search(text, match, pattern) || match.size() < 2) {
        return "";
    }

    return trim(match[1].str());
}

KernelCategory classify_kernel(const std::string& kernel_entry,
                               const std::string& primitive_id,
                               const std::string& primitive_type) {
    if (kernel_entry.rfind("reorder_weights_opt_", 0) == 0) {
        return KernelCategory::support_weight_reorder;
    }

    if (kernel_entry.rfind("reorder_data", 0) == 0 || primitive_type == "reorder") {
        if (primitive_id.rfind("result:", 0) == 0) {
            return KernelCategory::output;
        }
        return KernelCategory::support_reorder;
    }

    if (primitive_id.rfind("result:", 0) == 0) {
        return KernelCategory::output;
    }

    if (kernel_entry.empty()) {
        return KernelCategory::unknown;
    }

    if (primitive_id.rfind("constant:", 0) == 0 || primitive_id.rfind("data:", 0) == 0) {
        return KernelCategory::auxiliary;
    }

    return KernelCategory::main_compute;
}

std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) {
        return value;
    }

    std::string escaped = "\"";
    for (char ch : value) {
        if (ch == '"') {
            escaped += "\"\"";
        } else {
            escaped.push_back(ch);
        }
    }
    escaped.push_back('"');
    return escaped;
}

std::string markdown_escape(std::string value) {
    value = trim(value);
    if (value.empty()) {
        return "";
    }

    std::string escaped;
    escaped.reserve(value.size());
    for (char ch : value) {
        if (ch == '|') {
            escaped += "\\|";
        } else if (ch == '\r') {
            continue;
        } else if (ch == '\n') {
            escaped += "<br>";
        } else {
            escaped.push_back(ch);
        }
    }
    return escaped;
}

std::string join_strings(const std::vector<std::string>& values, const std::string& delimiter) {
    std::ostringstream output;
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            output << delimiter;
        }
        output << values[i];
    }
    return output.str();
}

std::string join_paths(const std::vector<std::filesystem::path>& paths, const std::string& delimiter) {
    std::vector<std::string> raw_paths;
    raw_paths.reserve(paths.size());
    for (const auto& path : paths) {
        raw_paths.push_back(path.string());
    }
    return join_strings(raw_paths, delimiter);
}

std::string join_markdown_lines(const std::vector<std::string>& values) {
    if (values.empty()) {
        return "";
    }

    std::vector<std::string> escaped_values;
    escaped_values.reserve(values.size());
    for (const auto& value : values) {
        escaped_values.push_back(markdown_escape(value));
    }

    return join_strings(escaped_values, "<br>");
}

std::string join_markdown_paths(const std::vector<std::filesystem::path>& paths) {
    std::vector<std::string> raw_paths;
    raw_paths.reserve(paths.size());
    for (const auto& path : paths) {
        raw_paths.push_back(path.string());
    }

    return join_markdown_lines(raw_paths);
}

std::string join_markdown_inline(const std::vector<std::string>& values) {
    if (values.empty()) {
        return "";
    }

    std::vector<std::string> escaped_values;
    escaped_values.reserve(values.size());
    for (const auto& value : values) {
        escaped_values.push_back(markdown_escape(value));
    }

    return join_strings(escaped_values, ", ");
}

std::vector<std::filesystem::path> unique_paths(std::vector<std::filesystem::path> paths) {
    std::sort(paths.begin(), paths.end());
    paths.erase(std::unique(paths.begin(), paths.end()), paths.end());
    return paths;
}

std::vector<std::string> unique_strings(std::vector<std::string> values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
    return values;
}

template <typename T>
void append_unique(std::vector<T>& values, const T& value) {
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}

//First Join logic?
std::string summary_key(const CorrelationOccurrenceRow& row) {
    if (!row.build_record.has_value()) {
        return row.occurrence.name + "|unmatched";
    }

    const auto& record = *row.build_record;
    return record.kernel_entry + "|" + record.primitive_id + "|" + record.implementation + "|" + record.batch_hash;
}

}  // namespace

std::string to_string(KernelCategory category) {
    switch (category) {
    case KernelCategory::main_compute:
        return "main_compute";
    case KernelCategory::support_reorder:
        return "support_reorder";
    case KernelCategory::support_weight_reorder:
        return "support_weight_reorder";
    case KernelCategory::auxiliary:
        return "auxiliary";
    case KernelCategory::output:
        return "output";
    case KernelCategory::unknown:
    default:
        return "unknown";
    }
}

std::string to_string(PrimitiveMappingKind mapping_kind) {
    switch (mapping_kind) {
    case PrimitiveMappingKind::unique_primitive:
        return "unique_primitive";
    case PrimitiveMappingKind::shared_primitives:
        return "shared_primitives";
    case PrimitiveMappingKind::unmatched:
    default:
        return "unmatched";
    }
}

std::vector<std::string> default_focus_kernel_entries() {
    return {
        "reorder_data_11475626742907933301_0_0",
        "convolution_gpu_bfyx_to_bfyx_f16_4370654348647957117_0_0",
        "convolution_gpu_bfyx_f16_1x1_11301203768358881115_0_0",
        "reduce_gpu_b_fs_yx_fsv16_17577769709185652434_0_0",
        "fully_connected_gpu_bf_io_gemm_10554338203459921666_0_0",
        "reorder_data_fast_b1_12550024482704781809_0_0",
        "reorder_data_14584053691340162655_0_0",
    };
}

std::vector<CorrelationOccurrenceRow> filter_occurrence_rows_by_kernel_entries(
    const std::vector<CorrelationOccurrenceRow>& rows,
    const std::vector<std::string>& kernel_entries) {
    if (kernel_entries.empty()) {
        return rows;
    }

    const auto unique_kernel_entries = unique_strings(kernel_entries);
    const std::unordered_set<std::string> kernel_filter(unique_kernel_entries.begin(),
                                                        unique_kernel_entries.end());

    std::vector<CorrelationOccurrenceRow> filtered_rows;
    filtered_rows.reserve(rows.size());
    for (const auto& row : rows) {
        const auto kernel_entry = row.build_record.has_value() ? row.build_record->kernel_entry : row.occurrence.name;
        if (kernel_filter.find(kernel_entry) != kernel_filter.end()) {
            filtered_rows.push_back(row);
        }
    }

    return filtered_rows;
}

std::vector<KernelIdentitySummary> filter_identity_summaries_by_kernel_entries(
    const std::vector<KernelIdentitySummary>& summaries,
    const std::vector<std::string>& kernel_entries) {
    if (kernel_entries.empty()) {
        return summaries;
    }

    const auto unique_kernel_entries = unique_strings(kernel_entries);
    const std::unordered_set<std::string> kernel_filter(unique_kernel_entries.begin(),
                                                        unique_kernel_entries.end());

    std::vector<KernelIdentitySummary> filtered_summaries;
    filtered_summaries.reserve(summaries.size());
    for (const auto& summary : summaries) {
        if (kernel_filter.find(summary.kernel_entry) != kernel_filter.end()) {
            filtered_summaries.push_back(summary);
        }
    }

    return filtered_summaries;
}

std::vector<GTPinKernelOccurrence> parse_gtpin_profile(const std::filesystem::path& profile_path) {
    std::ifstream input(profile_path);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open GTPin profile: " + profile_path.string());
    }

    std::vector<GTPinKernelOccurrence> occurrences;
    std::unordered_map<std::string, size_t> occurrence_counts;

    std::string line;
    size_t line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const auto trimmed_line = trim(line);
        if (trimmed_line.empty() || trimmed_line.rfind("###", 0) == 0 ||
            trimmed_line.rfind("Legend:", 0) == 0 || trimmed_line.rfind("NA -", 0) == 0 ||
            trimmed_line.rfind("Name", 0) == 0) {
            continue;
        }

        std::istringstream row_stream(line);
        GTPinKernelOccurrence occurrence;
        std::string simd;
        std::string frequency;
        std::string total_cycles;
        std::string average_cycles;
        std::string skipped;

        if (!(row_stream >> occurrence.name >> occurrence.hash_id >> simd >> occurrence.type >> frequency >>
              total_cycles >> average_cycles >> skipped >> occurrence.platform)) {
            continue;
        }

        std::getline(row_stream, occurrence.execution_descriptor);
        occurrence.execution_descriptor = trim(occurrence.execution_descriptor);

        if (occurrence.name == "igc_check") {
            continue;
        }

        occurrence.simd = parse_optional_u32(simd);
        occurrence.frequency = parse_optional_u64(frequency);
        occurrence.total_cycles = parse_optional_u64(total_cycles);
        occurrence.average_cycles = parse_optional_u64(average_cycles);
        occurrence.skipped = parse_optional_u64(skipped);
        occurrence.file_line = line_number;
        occurrence.occurrence_index = ++occurrence_counts[occurrence.name];

        occurrences.push_back(std::move(occurrence));
    }

    return occurrences;
}

std::vector<BuildImplementationRecord> parse_build_implementations_info(const std::filesystem::path& info_path) {
    const auto content = read_text_file(info_path);
    const auto blocks = split_blocks_by_top_level_braces(content);

    const std::regex primitive_id_regex(R"(\bid\s*:\s*([^,\r\n]+),)");
    const std::regex primitive_type_regex(R"(\btype\s*:\s*([^,\r\n]+),)");
    const std::regex constant_regex(R"(\bconstant\s*:\s*(true|false),)");
    const std::regex output_regex(R"(\boutput\s*:\s*(true|false),)");
    const std::regex in_data_flow_regex(R"(\bin data flow\s*:\s*(true|false),)");
    const std::regex implementation_regex(R"(\bimplementation\s*:\s*([^,\r\n]+),)");
    const std::regex in_shape_regex(R"(\bin_shape_of_subgraph\s*:\s*([^,\r\n]+),)");
    const std::regex batch_hash_regex(R"(batch_hash\s*:\s*([^,\r\n]*),)");
    const std::regex kernel_entry_regex(R"(kernel_entry\s*:\s*([^,\r\n]*),)");

    std::vector<BuildImplementationRecord> records;
    for (const auto& block : blocks) {
        BuildImplementationRecord record;
        record.primitive_id = extract_first_match(block, primitive_id_regex);
        record.primitive_type = extract_first_match(block, primitive_type_regex);
        record.implementation = extract_first_match(block, implementation_regex);
        record.batch_hash = extract_first_match(block, batch_hash_regex);
        record.kernel_entry = extract_first_match(block, kernel_entry_regex);
        record.source_info_file = info_path.string();

        const auto constant_value = extract_first_match(block, constant_regex);
        if (!constant_value.empty()) {
            record.constant = parse_bool_value(constant_value);
        }

        const auto output_value = extract_first_match(block, output_regex);
        if (!output_value.empty()) {
            record.output = parse_bool_value(output_value);
        }

        const auto in_data_flow_value = extract_first_match(block, in_data_flow_regex);
        if (!in_data_flow_value.empty()) {
            record.in_data_flow = parse_bool_value(in_data_flow_value);
        }

        const auto in_shape_value = extract_first_match(block, in_shape_regex);
        record.in_shape_of_subgraph = parse_optional_u32(in_shape_value);

        const bool has_kernel_identity = !record.kernel_entry.empty() &&
                                         !record.batch_hash.empty() &&
                                         !record.implementation.empty() &&
                                         record.implementation != "null";

        if (!has_kernel_identity) {
            continue;
        }

        records.push_back(std::move(record));
    }

    return records;
}

std::vector<SourceBucketRecord> parse_source_bucket_files(const std::vector<std::filesystem::path>& source_bucket_paths) {
    const std::regex filename_regex(
        R"(clDNN_program_(\d+)_bucket_(\d+)_part_(\d+)_([0-9]+)\.cl$)",
        std::regex_constants::icase);
    const std::regex kernel_name_regex(R"(// Kernel name:\s*(\S+))");

    std::vector<SourceBucketRecord> records;
    records.reserve(source_bucket_paths.size());

    for (const auto& path : source_bucket_paths) {
        SourceBucketRecord record;
        record.source_path = path;

        std::smatch filename_match;
        const auto filename = path.filename().string();
        if (std::regex_search(filename, filename_match, filename_regex) && filename_match.size() == 5) {
            record.program_id = parse_optional_u32(filename_match[1].str());
            record.bucket_id = parse_optional_u32(filename_match[2].str());
            record.part_id = parse_optional_u32(filename_match[3].str());
            record.batch_hash = trim(filename_match[4].str());
        }

        std::ifstream input(path);
        if (!input.is_open()) {
            throw std::runtime_error("Failed to open source bucket: " + path.string());
        }

        std::string line;
        while (std::getline(input, line)) {
            std::smatch match;
            if (std::regex_search(line, match, kernel_name_regex) && match.size() == 2) {
                append_unique(record.kernel_entries, trim(match[1].str()));
            }
        }

        records.push_back(std::move(record));
    }

    return records;
}

OfflineCorrelationArtifacts build_offline_correlation(const OfflineCorrelationOptions& options) {
    OfflineCorrelationArtifacts artifacts;
    artifacts.gtpin_occurrences = parse_gtpin_profile(options.gtpin_profile_path);

    for (const auto& build_info_path : options.build_info_paths) {
        auto parsed_records = parse_build_implementations_info(build_info_path);
        artifacts.build_records.insert(artifacts.build_records.end(),
                                       std::make_move_iterator(parsed_records.begin()),
                                       std::make_move_iterator(parsed_records.end()));
    }

    artifacts.source_buckets = parse_source_bucket_files(options.source_bucket_paths);

    std::unordered_map<std::string, std::vector<const BuildImplementationRecord*>> build_records_by_kernel_entry;
    for (const auto& record : artifacts.build_records) {
        build_records_by_kernel_entry[record.kernel_entry].push_back(&record);
    }

    std::unordered_map<std::string, std::vector<std::string>> primitive_ids_by_kernel_entry;
    for (const auto& [kernel_entry, records] : build_records_by_kernel_entry) {
        std::vector<std::string> primitive_ids;
        primitive_ids.reserve(records.size());
        for (const auto* record : records) {
            append_unique(primitive_ids, record->primitive_id);
        }
        primitive_ids_by_kernel_entry.emplace(kernel_entry, unique_strings(std::move(primitive_ids)));
    }

    std::unordered_map<std::string, std::vector<const SourceBucketRecord*>> source_buckets_by_batch_hash;
    std::unordered_map<std::string, std::vector<const SourceBucketRecord*>> source_buckets_by_kernel_entry;
    for (const auto& bucket : artifacts.source_buckets) {
        source_buckets_by_batch_hash[bucket.batch_hash].push_back(&bucket);
        for (const auto& kernel_entry : bucket.kernel_entries) {
            source_buckets_by_kernel_entry[kernel_entry].push_back(&bucket);
        }
    }

    for (const auto& occurrence : artifacts.gtpin_occurrences) {
        const auto build_it = build_records_by_kernel_entry.find(occurrence.name);
        if (build_it == build_records_by_kernel_entry.end()) {
            CorrelationOccurrenceRow row;
            row.occurrence = occurrence;
            row.category = KernelCategory::unknown;
            row.mapping_kind = PrimitiveMappingKind::unmatched;
            artifacts.occurrence_rows.push_back(std::move(row));
            continue;
        }

        const auto primitive_id_it = primitive_ids_by_kernel_entry.find(occurrence.name);
        const auto mapped_primitive_ids = primitive_id_it != primitive_ids_by_kernel_entry.end()
            ? primitive_id_it->second
            : std::vector<std::string>{};
        const auto primitive_fanout = mapped_primitive_ids.size();
        const auto mapping_kind = primitive_fanout == 0
            ? PrimitiveMappingKind::unmatched
            : (primitive_fanout == 1 ? PrimitiveMappingKind::unique_primitive
                                     : PrimitiveMappingKind::shared_primitives);

        for (const auto* build_record : build_it->second) {
            CorrelationOccurrenceRow row;
            row.occurrence = occurrence;
            row.build_record = *build_record;
            row.category = classify_kernel(build_record->kernel_entry,
                                           build_record->primitive_id,
                                           build_record->primitive_type);
            row.mapping_kind = mapping_kind;
            row.primitive_fanout = primitive_fanout;
            row.mapped_primitive_ids = mapped_primitive_ids;

            const auto bucket_it = source_buckets_by_batch_hash.find(build_record->batch_hash);
            if (bucket_it != source_buckets_by_batch_hash.end()) {
                for (const auto* bucket : bucket_it->second) {
                    if (bucket->kernel_entries.empty() ||
                        std::find(bucket->kernel_entries.begin(),
                                  bucket->kernel_entries.end(),
                                  build_record->kernel_entry) != bucket->kernel_entries.end()) {
                        row.source_paths.push_back(bucket->source_path);
                    }
                }
            }

            if (row.source_paths.empty()) {
                const auto kernel_source_it = source_buckets_by_kernel_entry.find(build_record->kernel_entry);
                if (kernel_source_it != source_buckets_by_kernel_entry.end()) {
                    for (const auto* bucket : kernel_source_it->second) {
                        row.source_paths.push_back(bucket->source_path);
                    }
                }
            }

            row.source_paths = unique_paths(std::move(row.source_paths));
            artifacts.occurrence_rows.push_back(std::move(row));
        }
    }

    std::unordered_map<std::string, KernelIdentitySummary> summary_by_key;
    for (const auto& row : artifacts.occurrence_rows) {
        const auto key = summary_key(row);
        auto& summary = summary_by_key[key];

        if (summary.kernel_entry.empty() && row.build_record.has_value()) {
            const auto& build_record = *row.build_record;
            summary.kernel_entry = build_record.kernel_entry;
            summary.primitive_id = build_record.primitive_id;
            summary.primitive_type = build_record.primitive_type;
            summary.implementation = build_record.implementation;
            summary.batch_hash = build_record.batch_hash;
            summary.category = row.category;
            summary.mapping_kind = row.mapping_kind;
            summary.primitive_fanout = row.primitive_fanout;
            summary.mapped_primitive_ids = row.mapped_primitive_ids;
        } else if (summary.kernel_entry.empty()) {
            summary.kernel_entry = row.occurrence.name;
            summary.category = row.category;
            summary.mapping_kind = row.mapping_kind;
            summary.primitive_fanout = row.primitive_fanout;
            summary.mapped_primitive_ids = row.mapped_primitive_ids;
        }

        summary.occurrence_count++;
        append_unique(summary.gtpin_names, row.occurrence.name);
        append_unique(summary.hash_ids, row.occurrence.hash_id);
        append_unique(summary.execution_descriptors, row.occurrence.execution_descriptor);
        for (const auto& path : row.source_paths) {
            append_unique(summary.source_paths, path);
        }
    }

    artifacts.identity_summaries.reserve(summary_by_key.size());
    for (auto& [unused_key, summary] : summary_by_key) {
        summary.source_paths = unique_paths(std::move(summary.source_paths));
        artifacts.identity_summaries.push_back(std::move(summary));
    }

    std::sort(artifacts.identity_summaries.begin(),
              artifacts.identity_summaries.end(),
              [](const KernelIdentitySummary& lhs, const KernelIdentitySummary& rhs) {
                  return lhs.kernel_entry < rhs.kernel_entry;
              });

    if (options.kernel_occurrences_csv_path.has_value()) {
        write_kernel_occurrences_csv(artifacts.occurrence_rows, *options.kernel_occurrences_csv_path);
    }

    if (options.kernel_identity_summary_csv_path.has_value()) {
        write_kernel_identity_summary_csv(artifacts.identity_summaries,
                                          *options.kernel_identity_summary_csv_path);
    }

    return artifacts;
}

void write_kernel_occurrences_csv(const std::vector<CorrelationOccurrenceRow>& rows,
                                  const std::filesystem::path& csv_path) {
    std::ofstream output(csv_path);
    if (!output.is_open()) {
        throw std::runtime_error("Failed to open occurrence CSV output: " + csv_path.string());
    }

    output << "gtpin_name,occurrence_index,file_line,hash_id,simd,type,frequency,total_cycles,average_cycles,"
              "skipped,platform,execution_descriptor,primitive_id,primitive_type,implementation,kernel_entry,"
              "batch_hash,source_paths,kernel_category,mapping_kind,primitive_fanout,mapped_primitive_ids\n";

    for (const auto& row : rows) {
        const auto& occurrence = row.occurrence;
        const auto primitive_id = row.build_record.has_value() ? row.build_record->primitive_id : "";
        const auto primitive_type = row.build_record.has_value() ? row.build_record->primitive_type : "";
        const auto implementation = row.build_record.has_value() ? row.build_record->implementation : "";
        const auto kernel_entry = row.build_record.has_value() ? row.build_record->kernel_entry : "";
        const auto batch_hash = row.build_record.has_value() ? row.build_record->batch_hash : "";

        output << csv_escape(occurrence.name) << ','
               << occurrence.occurrence_index << ','
               << occurrence.file_line << ','
               << csv_escape(occurrence.hash_id) << ','
               << csv_escape(occurrence.simd.has_value() ? std::to_string(*occurrence.simd) : "") << ','
               << csv_escape(occurrence.type) << ','
               << csv_escape(occurrence.frequency.has_value() ? std::to_string(*occurrence.frequency) : "") << ','
               << csv_escape(occurrence.total_cycles.has_value() ? std::to_string(*occurrence.total_cycles) : "") << ','
               << csv_escape(occurrence.average_cycles.has_value() ? std::to_string(*occurrence.average_cycles) : "") << ','
               << csv_escape(occurrence.skipped.has_value() ? std::to_string(*occurrence.skipped) : "") << ','
               << csv_escape(occurrence.platform) << ','
               << csv_escape(occurrence.execution_descriptor) << ','
               << csv_escape(primitive_id) << ','
               << csv_escape(primitive_type) << ','
               << csv_escape(implementation) << ','
               << csv_escape(kernel_entry) << ','
               << csv_escape(batch_hash) << ','
               << csv_escape(join_paths(row.source_paths, ";")) << ','
               << csv_escape(to_string(row.category)) << ','
               << csv_escape(to_string(row.mapping_kind)) << ','
               << row.primitive_fanout << ','
               << csv_escape(join_strings(row.mapped_primitive_ids, ";"))
               << '\n';
    }
}

void write_kernel_identity_summary_csv(const std::vector<KernelIdentitySummary>& summaries,
                                       const std::filesystem::path& csv_path) {
    std::ofstream output(csv_path);
    if (!output.is_open()) {
        throw std::runtime_error("Failed to open identity CSV output: " + csv_path.string());
    }

    output << "kernel_entry,primitive_id,primitive_type,implementation,batch_hash,source_paths,"
              "kernel_category,mapping_kind,primitive_fanout,mapped_primitive_ids,occurrence_count,gtpin_names,hash_ids,execution_descriptors\n";

    for (const auto& summary : summaries) {
        output << csv_escape(summary.kernel_entry) << ','
               << csv_escape(summary.primitive_id) << ','
               << csv_escape(summary.primitive_type) << ','
               << csv_escape(summary.implementation) << ','
               << csv_escape(summary.batch_hash) << ','
               << csv_escape(join_paths(summary.source_paths, ";")) << ','
               << csv_escape(to_string(summary.category)) << ','
               << csv_escape(to_string(summary.mapping_kind)) << ','
               << summary.primitive_fanout << ','
               << csv_escape(join_strings(summary.mapped_primitive_ids, ";")) << ','
               << summary.occurrence_count << ','
               << csv_escape(join_strings(summary.gtpin_names, ";")) << ','
               << csv_escape(join_strings(summary.hash_ids, ";")) << ','
               << csv_escape(join_strings(summary.execution_descriptors, ";"))
               << '\n';
    }
}

void write_kernel_occurrences_markdown(const std::vector<CorrelationOccurrenceRow>& rows,
                                       const std::filesystem::path& markdown_path) {
    std::ofstream output(markdown_path);
    if (!output.is_open()) {
        throw std::runtime_error("Failed to open occurrence markdown output: " + markdown_path.string());
    }

    output << "# Kernel Occurrences\n\n";
    output << "Compact occurrence-level view with the highest-signal correlation fields.\n\n";

    for (const auto& row : rows) {
        const auto primitive_id = row.build_record.has_value() ? row.build_record->primitive_id : "";
        const auto kernel_entry = row.build_record.has_value() ? row.build_record->kernel_entry : row.occurrence.name;
        const auto average_cycles = row.occurrence.average_cycles.has_value()
            ? std::to_string(*row.occurrence.average_cycles)
            : "";
        const auto simd = row.occurrence.simd.has_value()
            ? std::to_string(*row.occurrence.simd)
            : "";

        output << "- **Kernel Entry:** `" << markdown_escape(kernel_entry) << "`\n";
        output << "  **Primitive:** `" << markdown_escape(primitive_id) << "`  ";
        output << "**Category:** `" << markdown_escape(to_string(row.category)) << "`  ";
        output << "**Mapping:** `" << markdown_escape(to_string(row.mapping_kind)) << "`  ";
        output << "**Occurrence:** `" << row.occurrence.occurrence_index << "`  ";
        output << "**Avg Cycles:** `" << markdown_escape(average_cycles) << "`  ";
        output << "**SIMD:** `" << markdown_escape(simd) << "`\n\n";
    }
}

void write_kernel_identity_summary_markdown(const std::vector<KernelIdentitySummary>& summaries,
                                            const std::filesystem::path& markdown_path) {
    std::ofstream output(markdown_path);
    if (!output.is_open()) {
        throw std::runtime_error("Failed to open identity markdown output: " + markdown_path.string());
    }

    output << "# Kernel Identity Summary\n\n";
    output << "Grouped by `kernel_entry | primitive_id | implementation | batch_hash`.\n\n";

    for (const auto& summary : summaries) {
        output << "- **Kernel Entry:** `" << markdown_escape(summary.kernel_entry) << "`\n";
        output << "  **Primitive:** `" << markdown_escape(summary.primitive_id) << "`  ";
        output << "**Impl:** `" << markdown_escape(summary.implementation) << "`  ";
        output << "**Batch Hash:** `" << markdown_escape(summary.batch_hash) << "`\n";
        output << "  **Category:** `" << markdown_escape(to_string(summary.category)) << "`  ";
        output << "**Mapping:** `" << markdown_escape(to_string(summary.mapping_kind)) << "`  ";
        output << "**Fanout:** `" << summary.primitive_fanout << "`  ";
        output << "**Occurrences:** `" << summary.occurrence_count << "`";
        if (!summary.gtpin_names.empty()) {
            output << "  **GTPin Names:** `" << join_markdown_inline(summary.gtpin_names) << "`";
        }
        if (summary.mapping_kind == PrimitiveMappingKind::shared_primitives && !summary.mapped_primitive_ids.empty()) {
            output << "\n";
            output << "  **Mapped Primitive Set:** `" << join_markdown_inline(summary.mapped_primitive_ids) << "`";
        }
        output << "\n\n";
    }
}

}  // namespace ov::intel_gpu::gtpin
