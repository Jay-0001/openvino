#include "offline_correlation.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ov::intel_gpu::gtpin::OfflineCorrelationOptions;

struct CliOptions {
    std::filesystem::path gtpin_profile_path;
    std::vector<std::filesystem::path> build_info_paths;
    std::vector<std::filesystem::path> source_bucket_paths;
    std::filesystem::path output_dir;
    std::vector<std::string> kernel_entries;
    bool use_all_kernels = false;
};

std::string normalize_flag(const std::string& flag) {
    if (flag == "--build-info-dir") {
        return "--build-info-root";
    }
    if (flag == "--source-dir") {
        return "--source-root";
    }
    return flag;
}

std::filesystem::path require_value_path(int& index, int argc, char** argv, const std::string& flag) {
    if (index + 1 >= argc) {
        throw std::runtime_error("Missing value for argument: " + flag);
    }
    ++index;
    return std::filesystem::path(argv[index]);
}

std::string require_value_string(int& index, int argc, char** argv, const std::string& flag) {
    if (index + 1 >= argc) {
        throw std::runtime_error("Missing value for argument: " + flag);
    }
    ++index;
    return argv[index];
}

void append_matching_files(const std::filesystem::path& root,
                           const std::string& extension,
                           std::vector<std::filesystem::path>& output) {
    if (!std::filesystem::exists(root)) {
        throw std::runtime_error("Path does not exist: " + root.string());
    }

    if (std::filesystem::is_regular_file(root)) {
        if (root.extension() == extension) {
            output.push_back(root);
        }
        return;
    }

    for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
        if (entry.is_regular_file() && entry.path().extension() == extension) {
            output.push_back(entry.path());
        }
    }
}

template <typename T>
void sort_and_unique(std::vector<T>& values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

CliOptions parse_cli(int argc, char** argv) {
    CliOptions options;

    for (int i = 1; i < argc; ++i) {
        const auto flag = normalize_flag(argv[i]);

        if (flag == "--gtpin-profile") {
            options.gtpin_profile_path = require_value_path(i, argc, argv, flag);
        } else if (flag == "--build-info") {
            options.build_info_paths.push_back(require_value_path(i, argc, argv, flag));
        } else if (flag == "--build-info-root") {
            append_matching_files(require_value_path(i, argc, argv, flag), ".info", options.build_info_paths);
        } else if (flag == "--source-bucket") {
            options.source_bucket_paths.push_back(require_value_path(i, argc, argv, flag));
        } else if (flag == "--source-root") {
            append_matching_files(require_value_path(i, argc, argv, flag), ".cl", options.source_bucket_paths);
        } else if (flag == "--output-dir") {
            options.output_dir = require_value_path(i, argc, argv, flag);
        } else if (flag == "--kernel") {
            options.kernel_entries.push_back(require_value_string(i, argc, argv, flag));
        } else if (flag == "--all-kernels") {
            options.use_all_kernels = true;
        } else if (flag == "--help" || flag == "-h") {
            throw std::runtime_error(
                "Usage: offline_correlation --gtpin-profile <file> "
                "[--build-info <file> | --build-info-root <dir>]... "
                "[--source-bucket <file> | --source-root <dir>]... "
                "--output-dir <dir> [--kernel <kernel_entry>]... [--all-kernels]");
        } else {
            throw std::runtime_error("Unknown argument: " + flag);
        }
    }

    if (options.gtpin_profile_path.empty()) {
        throw std::runtime_error("Missing required argument: --gtpin-profile");
    }

    if (options.build_info_paths.empty()) {
        throw std::runtime_error("At least one build_implementations.info path is required.");
    }

    if (options.source_bucket_paths.empty()) {
        throw std::runtime_error("At least one source bucket path is required.");
    }

    if (options.output_dir.empty()) {
        throw std::runtime_error("Missing required argument: --output-dir");
    }

    sort_and_unique(options.build_info_paths);
    sort_and_unique(options.source_bucket_paths);
    sort_and_unique(options.kernel_entries);
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto cli = parse_cli(argc, argv);

        std::filesystem::create_directories(cli.output_dir);

        OfflineCorrelationOptions options;
        options.gtpin_profile_path = cli.gtpin_profile_path;
        options.build_info_paths = cli.build_info_paths;
        options.source_bucket_paths = cli.source_bucket_paths;

        const auto artifacts = ov::intel_gpu::gtpin::build_offline_correlation(options);

        std::vector<std::string> active_kernel_entries;
        if (!cli.use_all_kernels) {
            active_kernel_entries = cli.kernel_entries.empty()
                ? ov::intel_gpu::gtpin::default_focus_kernel_entries()
                : cli.kernel_entries;
        }

        const auto filtered_rows = ov::intel_gpu::gtpin::filter_occurrence_rows_by_kernel_entries(
            artifacts.occurrence_rows,
            active_kernel_entries);
        const auto filtered_summaries = ov::intel_gpu::gtpin::filter_identity_summaries_by_kernel_entries(
            artifacts.identity_summaries,
            active_kernel_entries);

        const auto occurrences_csv = cli.output_dir / "kernel_occurrences.csv";
        const auto summary_csv = cli.output_dir / "kernel_identity_summary.csv";
        const auto occurrences_markdown = cli.output_dir / "kernel_occurrences.md";
        const auto summary_markdown = cli.output_dir / "kernel_identity_summary.md";

        ov::intel_gpu::gtpin::write_kernel_occurrences_csv(filtered_rows, occurrences_csv);
        ov::intel_gpu::gtpin::write_kernel_identity_summary_csv(filtered_summaries, summary_csv);
        ov::intel_gpu::gtpin::write_kernel_occurrences_markdown(filtered_rows, occurrences_markdown);
        ov::intel_gpu::gtpin::write_kernel_identity_summary_markdown(filtered_summaries, summary_markdown);

        std::cout << "Offline correlation complete.\n";
        std::cout << "Kernel occurrences CSV: " << occurrences_csv.string() << '\n';
        std::cout << "Kernel identity summary CSV: " << summary_csv.string() << '\n';
        std::cout << "Kernel occurrences Markdown: " << occurrences_markdown.string() << '\n';
        std::cout << "Kernel identity summary Markdown: " << summary_markdown.string() << '\n';
        std::cout << "Occurrence rows written: " << filtered_rows.size() << '\n';
        std::cout << "Identity rows written: " << filtered_summaries.size() << '\n';

        if (cli.use_all_kernels) {
            std::cout << "Selection mode: full dataset\n";
        } else if (!cli.kernel_entries.empty()) {
            std::cout << "Selection mode: explicit kernel_entry filter (" << cli.kernel_entries.size() << " kernels)\n";
        } else {
            std::cout << "Selection mode: default focused kernel slice (7 kernels)\n";
        }

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
