param(
    [Parameter(Mandatory = $true)]
    [string]$DispatchDir,

    [Parameter(Mandatory = $true)]
    [string]$OfflineSummaryDir,

    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $OfflineSummaryDir "dispatch_correlation_analysis"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$dispatchFiles = Get-ChildItem -Path $DispatchDir -Filter "dispatch_map_raw*.csv" | Sort-Object Name
if ($dispatchFiles.Count -eq 0) {
    throw "No dispatch_map_raw*.csv files found in $DispatchDir"
}

$occurrencesPath = Join-Path $OfflineSummaryDir "kernel_occurrences.csv"
$identityPath = Join-Path $OfflineSummaryDir "kernel_identity_summary.csv"

if (-not (Test-Path -LiteralPath $occurrencesPath)) {
    throw "Missing kernel_occurrences.csv in $OfflineSummaryDir"
}

if (-not (Test-Path -LiteralPath $identityPath)) {
    throw "Missing kernel_identity_summary.csv in $OfflineSummaryDir"
}

function Convert-ToInt {
    param([string]$Value)
    if ([string]::IsNullOrWhiteSpace($Value)) {
        return 0
    }
    return [int]$Value
}

$dispatchRows = @()
foreach ($file in $dispatchFiles) {
    $rows = Import-Csv -LiteralPath $file.FullName
    foreach ($row in $rows) {
        $row | Add-Member -NotePropertyName source_file -NotePropertyValue $file.Name
        $row | Add-Member -NotePropertyName net_id_num -NotePropertyValue (Convert-ToInt $row.net_id)
        $row | Add-Member -NotePropertyName iteration_num -NotePropertyValue (Convert-ToInt $row.iteration)
        $row | Add-Member -NotePropertyName exec_index_num -NotePropertyValue (Convert-ToInt $row.exec_index)
        $dispatchRows += $row
    }
}

$dispatchRowsWithKernel = $dispatchRows | Where-Object { -not [string]::IsNullOrWhiteSpace($_.kernel_entry) }
$occurrenceRows = Import-Csv -LiteralPath $occurrencesPath
$identityRows = Import-Csv -LiteralPath $identityPath

$identityByKernel = @{}
foreach ($row in $identityRows) {
    if (-not $identityByKernel.ContainsKey($row.kernel_entry)) {
        $identityByKernel[$row.kernel_entry] = $row
    }
}

$occurrencesByKernel = @{}
foreach ($row in $occurrenceRows) {
    if (-not $occurrencesByKernel.ContainsKey($row.kernel_entry)) {
        $occurrencesByKernel[$row.kernel_entry] = New-Object System.Collections.ArrayList
    }
    $row | Add-Member -NotePropertyName occurrence_index_num -NotePropertyValue (Convert-ToInt $row.occurrence_index)
    [void]$occurrencesByKernel[$row.kernel_entry].Add($row)
}

$dispatchByKernel = @{}
foreach ($row in $dispatchRowsWithKernel) {
    if (-not $dispatchByKernel.ContainsKey($row.kernel_entry)) {
        $dispatchByKernel[$row.kernel_entry] = New-Object System.Collections.ArrayList
    }
    [void]$dispatchByKernel[$row.kernel_entry].Add($row)
}

$sufficiencyRows = @()
$candidateRows = @()

foreach ($kernelEntry in ($identityByKernel.Keys | Sort-Object)) {
    $identity = $identityByKernel[$kernelEntry]
    $expectedOccurrences = Convert-ToInt $identity.occurrence_count
    $dispatchGroup = @()
    if ($dispatchByKernel.ContainsKey($kernelEntry)) {
        $dispatchGroup = @($dispatchByKernel[$kernelEntry] | Sort-Object net_id_num, iteration_num, exec_index_num, primitive_id)
    }
    $occurrenceGroup = @()
    if ($occurrencesByKernel.ContainsKey($kernelEntry)) {
        $occurrenceGroup = @($occurrencesByKernel[$kernelEntry] | Sort-Object occurrence_index_num, file_line)
    }

    $dispatchCount = $dispatchGroup.Count
    $occurrenceCount = $occurrenceGroup.Count
    $countMatch = ($dispatchCount -eq $expectedOccurrences) -and ($occurrenceCount -eq $expectedOccurrences)

    $sufficiency = if ($countMatch) { "sufficient_for_order_proxy" } else { "insufficient_count_mismatch" }
    $reason = if ($countMatch) {
        "Dispatch rows and offline occurrences match in count; candidate occurrence-to-primitive mapping can be formed with (net_id, iteration, exec_index)."
    } elseif ($dispatchCount -eq 0) {
        "Kernel entry is absent from the dispatch dump."
    } elseif ($dispatchCount -lt $expectedOccurrences) {
        "Dispatch dump records fewer rows than GTPin occurrences; network-level primitive logging is undercounting true launches."
    } else {
        "Dispatch dump records more rows than expected occurrences; ordering cannot be trusted without deeper inspection."
    }

    $sufficiencyRows += [pscustomobject]@{
        kernel_entry = $kernelEntry
        mapping_kind = $identity.mapping_kind
        primitive_fanout = Convert-ToInt $identity.primitive_fanout
        expected_occurrences = $expectedOccurrences
        offline_occurrence_rows = $occurrenceCount
        dispatch_rows = $dispatchCount
        sufficiency = $sufficiency
        reason = $reason
        mapped_primitive_ids = $identity.mapped_primitive_ids
    }

    if ($countMatch) {
        for ($i = 0; $i -lt $expectedOccurrences; $i++) {
            $dispatchRow = $dispatchGroup[$i]
            $occurrenceRow = $occurrenceGroup[$i]
            $candidateRows += [pscustomobject]@{
                kernel_entry = $kernelEntry
                mapping_kind = $identity.mapping_kind
                occurrence_index = $occurrenceRow.occurrence_index
                gtpin_name = $occurrenceRow.gtpin_name
                candidate_primitive_id = $dispatchRow.primitive_id
                candidate_primitive_type = $dispatchRow.primitive_type
                implementation = $dispatchRow.implementation
                net_id = $dispatchRow.net_id
                iteration = $dispatchRow.iteration
                exec_index = $dispatchRow.exec_index
                source_file = $dispatchRow.source_file
            }
        }
    }
}

$sufficiencyCsv = Join-Path $OutputDir "dispatch_sufficiency_summary.csv"
$candidateCsv = Join-Path $OutputDir "dispatch_occurrence_candidates.csv"
$reportMd = Join-Path $OutputDir "dispatch_correlation_report.md"

$sufficiencyRows | Sort-Object kernel_entry | Export-Csv -NoTypeInformation -LiteralPath $sufficiencyCsv
$candidateRows | Sort-Object kernel_entry, {[int]$_.occurrence_index} | Export-Csv -NoTypeInformation -LiteralPath $candidateCsv

$totalKernels = $sufficiencyRows.Count
$sufficientCount = @($sufficiencyRows | Where-Object { $_.sufficiency -eq "sufficient_for_order_proxy" }).Count
$insufficientCount = $totalKernels - $sufficientCount
$sharedSufficient = @($sufficiencyRows | Where-Object { $_.mapping_kind -eq "shared_primitives" -and $_.sufficiency -eq "sufficient_for_order_proxy" }).Count
$sharedInsufficient = @($sufficiencyRows | Where-Object { $_.mapping_kind -eq "shared_primitives" -and $_.sufficiency -ne "sufficient_for_order_proxy" }).Count

$fileSummary = $dispatchRows |
    Group-Object source_file |
    Sort-Object Name |
    ForEach-Object {
        $group = $_.Group
        [pscustomobject]@{
            source_file = $_.Name
            net_ids = (($group.net_id | Select-Object -Unique) -join ",")
            rows = $_.Count
            kernel_rows = @($group | Where-Object { -not [string]::IsNullOrWhiteSpace($_.kernel_entry) }).Count
        }
    }

$mismatchExamples = $sufficiencyRows |
    Where-Object { $_.sufficiency -ne "sufficient_for_order_proxy" } |
    Sort-Object kernel_entry |
    Select-Object -First 12

$report = New-Object System.Collections.Generic.List[string]
$report.Add("# Dispatch Correlation Report")
$report.Add("")
$report.Add("## Verdict")
$report.Add("")
$report.Add("- Total kernels assessed: $totalKernels")
$report.Add("- Kernels sufficient for order-proxy correlation: $sufficientCount")
$report.Add("- Kernels insufficient under current artifacts: $insufficientCount")
$report.Add("- Shared-kernel entries sufficient for order-proxy correlation: $sharedSufficient")
$report.Add("- Shared-kernel entries still insufficient: $sharedInsufficient")
$report.Add("")
$report.Add("Current conclusion: the present files are sufficient to form candidate occurrence-to-primitive mappings for the kernels whose dispatch-row counts match the offline occurrence counts. They are not sufficient as a complete end-to-end correlation solution because some support kernels, especially weight-reorder kernels, are undercounted by the network-level dump.")
$report.Add("")
$report.Add("## Dispatch Files")
$report.Add("")
foreach ($row in $fileSummary) {
    $report.Add("- $($row.source_file): net_id=$($row.net_ids), rows=$($row.rows), rows_with_kernel_entry=$($row.kernel_rows)")
}
$report.Add("")
$report.Add("## Why Local Dispatch Indices Still Work Here")
$report.Add("")
$report.Add("- `exec_index` is file-local and resets per network dump, so it is not a global dispatch id.")
$report.Add("- For this analyzer, candidate mappings use the tuple `(net_id, iteration, exec_index)` rather than `exec_index` alone.")
$report.Add("- That is good enough for per-network ordering checks, but it remains a proxy rather than a true global launch sequence.")
$report.Add("")
$report.Add("## Insufficient Cases")
$report.Add("")
if ($mismatchExamples.Count -eq 0) {
    $report.Add("- None")
} else {
    foreach ($row in $mismatchExamples) {
        $report.Add("- ``$($row.kernel_entry)``: expected=$($row.expected_occurrences), dispatch_rows=$($row.dispatch_rows), mapping=$($row.mapping_kind)")
        $report.Add("  Reason: $($row.reason)")
    }
}
$report.Add("")
$report.Add("## Generated Files")
$report.Add("")
$report.Add('- `dispatch_sufficiency_summary.csv`')
$report.Add('- `dispatch_occurrence_candidates.csv`')

Set-Content -LiteralPath $reportMd -Value $report -Encoding UTF8

Write-Output "Wrote:"
Write-Output "  $sufficiencyCsv"
Write-Output "  $candidateCsv"
Write-Output "  $reportMd"
