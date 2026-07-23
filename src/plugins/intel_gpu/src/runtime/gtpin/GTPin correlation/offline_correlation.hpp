#pragma once

#include "offline_correlation_types.hpp"

namespace ov::intel_gpu::gtpin {

OfflineCorrelationArtifacts build_offline_correlation(
    const OfflineCorrelationOptions& options);

std::vector<std::string> default_focus_kernel_entries();

std::vector<CorrelationOccurrenceRow> filter_occurrence_rows_by_kernel_entries(
    const std::vector<CorrelationOccurrenceRow>& rows,
    const std::vector<std::string>& kernel_entries);

std::vector<KernelIdentitySummary> filter_identity_summaries_by_kernel_entries(
    const std::vector<KernelIdentitySummary>& summaries,
    const std::vector<std::string>& kernel_entries);

void write_kernel_occurrences_csv(
    const std::vector<CorrelationOccurrenceRow>& rows,
    const std::filesystem::path& csv_path);

void write_kernel_identity_summary_csv(
    const std::vector<KernelIdentitySummary>& summaries,
    const std::filesystem::path& csv_path);

void write_kernel_occurrences_markdown(
    const std::vector<CorrelationOccurrenceRow>& rows,
    const std::filesystem::path& markdown_path);

void write_kernel_identity_summary_markdown(
    const std::vector<KernelIdentitySummary>& summaries,
    const std::filesystem::path& markdown_path);

}  // namespace ov::intel_gpu::gtpin
