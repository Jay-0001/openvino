param(
    [Parameter(Mandatory = $true)]
    [string]$DispatchCsv,

    [Parameter(Mandatory = $true)]
    [Alias("KernelArgDump", "GTPinProfile")]
    [string]$GtpinDump,

    [string]$OutputDir = "",

    [string]$ModelLabel = "unknown_model"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path (Split-Path -Parent $DispatchCsv) "multi_inference_robustness_analysis"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

if (-not (Test-Path -LiteralPath $DispatchCsv)) {
    throw "Missing dispatch csv: $DispatchCsv"
}

if (-not (Test-Path -LiteralPath $GtpinDump)) {
    throw "Missing GTPin dump: $GtpinDump"
}

function Convert-ToInt64 {
    param($Value)
    if ($null -eq $Value -or [string]::IsNullOrWhiteSpace([string]$Value)) {
        return [int64]0
    }

    return [int64]$Value
}

function Join-UniqueValues {
    param([object[]]$Values)

    if ($null -eq $Values -or $Values.Count -eq 0) {
        return ""
    }

    return (($Values |
        Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } |
        Sort-Object -Unique) -join ";")
}

function Get-AddressEntries {
    param([string]$Value)

    if ([string]::IsNullOrWhiteSpace($Value)) {
        return @()
    }

    return @($Value -split ";" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
}

function Get-AddressIdentity {
    param([string]$Entry)

    if ([string]::IsNullOrWhiteSpace($Entry)) {
        return ""
    }

    $parts = $Entry -split ":"
    if ($parts.Count -eq 0) {
        return ""
    }

    return $parts[$parts.Count - 1]
}

function Get-AddressSet {
    param([string]$Value)

    $set = New-Object 'System.Collections.Generic.HashSet[string]'
    foreach ($entry in (Get-AddressEntries $Value)) {
        $identity = Get-AddressIdentity $entry
        if (-not [string]::IsNullOrWhiteSpace($identity)) {
            [void]$set.Add($identity.ToUpper())
        }
    }

    return $set
}

function Get-GtpinPointerList {
    param([string[]]$Values)

    if ($null -eq $Values -or $Values.Count -eq 0) {
        return @()
    }

    return @($Values |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
        ForEach-Object { (($_ -replace '^0x', '').Trim()).ToUpper() })
}

function Test-PointerCoverage {
    param(
        [string[]]$Expected,
        [System.Collections.Generic.HashSet[string]]$ActualSet
    )

    foreach ($pointer in $Expected) {
        if (-not $ActualSet.Contains($pointer)) {
            return $false
        }
    }

    return $true
}

function Get-ExecutionUnitKey {
    param(
        [int64]$NetId,
        [int64]$Iteration
    )

    return ("n{0}_i{1}" -f $NetId, $Iteration)
}

function Parse-GtpinDispatchDump {
    param([string]$Path)

    $dispatches = New-Object System.Collections.Generic.List[object]
    $current = $null
    $currentRole = ""

    foreach ($line in Get-Content -LiteralPath $Path) {
        if ($line -match '^DispatchId:\s*(\d+)') {
            if ($null -ne $current) {
                $dispatches.Add([pscustomobject]$current)
            }

            $current = @{
                dispatch_id = [int64]$matches[1]
                kernel = ""
                unique_name = ""
                extended_name = ""
                execution_descriptor = ""
                pointer_values = New-Object System.Collections.Generic.List[string]
                input_pointer_values = New-Object System.Collections.Generic.List[string]
                output_pointer_values = New-Object System.Collections.Generic.List[string]
                arg_ordinals = New-Object System.Collections.Generic.List[string]
                raw_pointer_records = New-Object System.Collections.Generic.List[string]
            }
            $currentRole = ""
            continue
        }

        if ($null -eq $current) {
            continue
        }

        if ($line -match '^Kernel:\s*(.+)$') {
            $current.kernel = $matches[1].Trim()
            continue
        }

        if ($line -match '^UniqueName:\s*(.+)$') {
            $current.unique_name = $matches[1].Trim()
            continue
        }

        if ($line -match '^ExtendedName:\s*(.+)$') {
            $current.extended_name = $matches[1].Trim()
            continue
        }

        if ($line -match '^ExecutionDescriptor:\s*(.+)$') {
            $current.execution_descriptor = $matches[1].Trim()
            continue
        }

        if ($line -match '^ArgOrdinal=(\d+).*Type=arg_bypointer.*Role=([A-Z]+)') {
            $current.arg_ordinals.Add($matches[1])
            $currentRole = $matches[2]
            continue
        }

        if ($line -match '^\s*PointerValue=(0x[0-9A-Fa-f]+)') {
            $pointerValue = $matches[1]
            $ordinal = ""
            if ($current.arg_ordinals.Count -gt $current.pointer_values.Count) {
                $ordinal = $current.arg_ordinals[$current.pointer_values.Count]
            }

            $current.pointer_values.Add($pointerValue)
            $current.raw_pointer_records.Add(($ordinal + ":" + $pointerValue).TrimStart(":"))

            if ($currentRole -eq "INPUT") {
                $current.input_pointer_values.Add($pointerValue)
            } elseif ($currentRole -eq "OUTPUT" -or $currentRole -eq "INOUT") {
                $current.output_pointer_values.Add($pointerValue)
            }

            continue
        }
    }

    if ($null -ne $current) {
        $dispatches.Add([pscustomobject]$current)
    }

    return $dispatches
}

$dispatchRows = Import-Csv -LiteralPath $DispatchCsv | ForEach-Object {
    $globalDispatchId = if ($_.PSObject.Properties.Name -contains "global_dispatch_id") {
        Convert-ToInt64 $_.global_dispatch_id
    } else {
        Convert-ToInt64 $_.dispatch_index
    }

    $netId = Convert-ToInt64 $_.net_id
    $iteration = Convert-ToInt64 $_.iteration
    $inputAddressSet = Get-AddressSet $_.input_arg_addresses
    $outputAddressSet = Get-AddressSet $_.output_arg_addresses
    $outputMemorySet = Get-AddressSet $_.output_memory_addresses

    [pscustomobject]@{
        net_id = $_.net_id
        net_id_num = $netId
        iteration = $_.iteration
        iteration_num = $iteration
        execution_unit_key = Get-ExecutionUnitKey -NetId $netId -Iteration $iteration
        is_internal_network_candidate = if ($netId -eq 0) { "true" } else { "false" }
        dispatch_index = $_.dispatch_index
        dispatch_index_num = Convert-ToInt64 $_.dispatch_index
        global_dispatch_id = $globalDispatchId
        primitive_id = $_.primitive_id
        primitive_type = $_.primitive_type
        implementation = $_.implementation
        kernel_index = $_.kernel_index
        kernel_index_num = Convert-ToInt64 $_.kernel_index
        kernel_entry = $_.kernel_entry
        batch_hash = $_.batch_hash
        input_arg_addresses = $_.input_arg_addresses
        output_arg_addresses = if ($_.PSObject.Properties.Name -contains "output_arg_addresses") { $_.output_arg_addresses } else { "" }
        output_memory_addresses = if ($_.PSObject.Properties.Name -contains "output_memory_addresses") { $_.output_memory_addresses } else { "" }
        input_address_set = $inputAddressSet
        output_address_set = $outputAddressSet
        output_memory_set = $outputMemorySet
    }
} | Sort-Object global_dispatch_id

$dispatchRowsWithKernel = @($dispatchRows | Where-Object { -not [string]::IsNullOrWhiteSpace($_.kernel_entry) })
if ($dispatchRowsWithKernel.Count -eq 0) {
    throw "Dispatch csv has no rows with kernel_entry values: $DispatchCsv"
}

$gtpinRows = Parse-GtpinDispatchDump -Path $GtpinDump | Sort-Object dispatch_id
if ($gtpinRows.Count -eq 0) {
    throw "Parsed zero GTPin dispatch rows from $GtpinDump"
}

$ovByGlobalId = @{}
foreach ($row in $dispatchRowsWithKernel) {
    $ovByGlobalId[[string]$row.global_dispatch_id] = $row
}

$gtpinByDispatchId = @{}
foreach ($row in $gtpinRows) {
    $gtpinByDispatchId[[string]$row.dispatch_id] = $row
}

$joinedRows = New-Object System.Collections.Generic.List[object]
foreach ($gtpinRow in $gtpinRows) {
    $ovRow = $null
    if ($ovByGlobalId.ContainsKey([string]$gtpinRow.dispatch_id)) {
        $ovRow = $ovByGlobalId[[string]$gtpinRow.dispatch_id]
    }

    $ovInputPointers = @()
    $ovOutputPointers = @()
    $gtpinInputPointers = @(Get-GtpinPointerList $gtpinRow.input_pointer_values)
    $gtpinOutputPointers = @(Get-GtpinPointerList $gtpinRow.output_pointer_values)
    $gtpinAllPointers = @(Get-GtpinPointerList $gtpinRow.pointer_values)
    $gtpinAllPointerSet = [System.Collections.Generic.HashSet[string]]::new([string[]]$gtpinAllPointers)
    $pointerCoverage = "false"
    $kernelNameMatch = "false"

    if ($null -ne $ovRow) {
        $ovInputPointers = @(Get-AddressEntries $ovRow.input_arg_addresses | ForEach-Object { Get-AddressIdentity $_ } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | ForEach-Object { $_.ToUpper() })
        $ovOutputPointers = @(Get-AddressEntries $ovRow.output_arg_addresses | ForEach-Object { Get-AddressIdentity $_ } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | ForEach-Object { $_.ToUpper() })
        $inputCovered = Test-PointerCoverage -Expected $ovInputPointers -ActualSet $gtpinAllPointerSet
        $outputCovered = Test-PointerCoverage -Expected $ovOutputPointers -ActualSet $gtpinAllPointerSet
        $pointerCoverage = if ($inputCovered -and $outputCovered) { "true" } else { "false" }
        $kernelNameMatch = if ($ovRow.kernel_entry -eq $gtpinRow.kernel) { "true" } else { "false" }
    }

    $joinedRows.Add([pscustomobject]@{
        model_label = $ModelLabel
        dispatch_id = $gtpinRow.dispatch_id
        ov_match = if ($null -ne $ovRow) { "matched" } else { "missing_ov_dispatch" }
        net_id = if ($null -ne $ovRow) { $ovRow.net_id } else { "" }
        net_id_num = if ($null -ne $ovRow) { $ovRow.net_id_num } else { [int64]-1 }
        iteration = if ($null -ne $ovRow) { $ovRow.iteration } else { "" }
        iteration_num = if ($null -ne $ovRow) { $ovRow.iteration_num } else { [int64]-1 }
        execution_unit_key = if ($null -ne $ovRow) { $ovRow.execution_unit_key } else { "" }
        is_internal_network_candidate = if ($null -ne $ovRow) { $ovRow.is_internal_network_candidate } else { "false" }
        dispatch_index = if ($null -ne $ovRow) { $ovRow.dispatch_index } else { "" }
        primitive_id = if ($null -ne $ovRow) { $ovRow.primitive_id } else { "" }
        primitive_type = if ($null -ne $ovRow) { $ovRow.primitive_type } else { "" }
        implementation = if ($null -ne $ovRow) { $ovRow.implementation } else { "" }
        kernel_entry = if ($null -ne $ovRow) { $ovRow.kernel_entry } else { "" }
        gtpin_kernel = $gtpinRow.kernel
        gtpin_unique_name = $gtpinRow.unique_name
        gtpin_extended_name = $gtpinRow.extended_name
        execution_descriptor = $gtpinRow.execution_descriptor
        kernel_name_match = $kernelNameMatch
        pointer_coverage_match = $pointerCoverage
        gtpin_input_pointers = ($gtpinInputPointers -join ";")
        gtpin_output_pointers = ($gtpinOutputPointers -join ";")
        gtpin_all_pointers = ($gtpinAllPointers -join ";")
        gtpin_raw_pointer_records = ($gtpinRow.raw_pointer_records -join ";")
    }) | Out-Null
}

$ovOnlyRows = New-Object System.Collections.Generic.List[object]
foreach ($ovRow in $dispatchRowsWithKernel) {
    if (-not $gtpinByDispatchId.ContainsKey([string]$ovRow.global_dispatch_id)) {
        $ovOnlyRows.Add([pscustomobject]@{
            model_label = $ModelLabel
            dispatch_id = $ovRow.global_dispatch_id
            net_id = $ovRow.net_id
            iteration = $ovRow.iteration
            execution_unit_key = $ovRow.execution_unit_key
            primitive_id = $ovRow.primitive_id
            primitive_type = $ovRow.primitive_type
            implementation = $ovRow.implementation
            kernel_entry = $ovRow.kernel_entry
            reason = "missing_gtpin_dispatch"
        }) | Out-Null
    }
}

$executionUnitSummaryRows = New-Object System.Collections.Generic.List[object]
$executionUnitOrder = @{}
$steadyExecutionUnits = @(
    $dispatchRowsWithKernel |
    Where-Object { $_.net_id_num -ne 0 } |
    Group-Object execution_unit_key |
    ForEach-Object {
        $_.Group | Sort-Object global_dispatch_id | Select-Object -First 1
    } |
    Sort-Object global_dispatch_id
)

$logicalExecutionIndex = 0
foreach ($unit in $steadyExecutionUnits) {
    $executionUnitOrder[$unit.execution_unit_key] = $logicalExecutionIndex
    $logicalExecutionIndex++
}

$executionUnits = @(
    $dispatchRowsWithKernel |
    Group-Object execution_unit_key |
    Sort-Object { ($_.Group | Sort-Object global_dispatch_id | Select-Object -First 1).global_dispatch_id }
)

foreach ($unitGroup in $executionUnits) {
    $group = @($unitGroup.Group | Sort-Object global_dispatch_id)
    $first = $group[0]
    $matchedRowsInUnit = @($joinedRows | Where-Object { $_.ov_match -eq "matched" -and $_.execution_unit_key -eq $first.execution_unit_key })
    $missingGtpinRowsInUnit = @($ovOnlyRows | Where-Object { $_.execution_unit_key -eq $first.execution_unit_key })
    $logicalIndexValue = ""
    if ($executionUnitOrder.ContainsKey($first.execution_unit_key)) {
        $logicalIndexValue = $executionUnitOrder[$first.execution_unit_key]
    }

    $executionUnitSummaryRows.Add([pscustomobject]@{
        model_label = $ModelLabel
        execution_unit_key = $first.execution_unit_key
        net_id = $first.net_id
        iteration = $first.iteration
        is_internal_network_candidate = $first.is_internal_network_candidate
        logical_execution_index = $logicalIndexValue
        ov_dispatch_rows = $group.Count
        matched_gtpin_rows = $matchedRowsInUnit.Count
        missing_gtpin_rows = $missingGtpinRowsInUnit.Count
        distinct_ov_kernels = @($group | Select-Object -ExpandProperty kernel_entry -Unique).Count
        distinct_matched_gtpin_kernels = @($matchedRowsInUnit | Select-Object -ExpandProperty gtpin_kernel -Unique).Count
        first_dispatch_id = $first.global_dispatch_id
        last_dispatch_id = ($group | Select-Object -Last 1).global_dispatch_id
    }) | Out-Null
}

$networkSummaryRows = New-Object System.Collections.Generic.List[object]
$networkGroups = @(
    $dispatchRowsWithKernel |
    Group-Object net_id_num |
    Sort-Object Name
)

foreach ($networkGroup in $networkGroups) {
    $group = @($networkGroup.Group | Sort-Object global_dispatch_id)
    $netId = $group[0].net_id_num
    $unitRows = @($executionUnitSummaryRows | Where-Object { [int64]$_.net_id -eq $netId } | Sort-Object first_dispatch_id)
    $rowsPerUnit = @($unitRows | Select-Object -ExpandProperty ov_dispatch_rows)
    $interpretation = ""
    if ($netId -eq 0) {
        $interpretation = "internal_setup_network_candidate"
    } elseif ($unitRows.Count -gt 1) {
        $interpretation = "repeated_execution_network"
    } else {
        $interpretation = "single_execution_network"
    }

    $networkSummaryRows.Add([pscustomobject]@{
        model_label = $ModelLabel
        net_id = $group[0].net_id
        is_internal_network_candidate = if ($netId -eq 0) { "true" } else { "false" }
        execution_unit_count = $unitRows.Count
        rows_total = $group.Count
        rows_per_execution_unit = (($unitRows | ForEach-Object { "{0}:{1}" -f $_.iteration, $_.ov_dispatch_rows }) -join ";")
        distinct_row_counts_per_execution_unit = Join-UniqueValues @($rowsPerUnit)
        distinct_kernel_entries = @($group | Select-Object -ExpandProperty kernel_entry -Unique).Count
        first_dispatch_id = $group[0].global_dispatch_id
        last_dispatch_id = ($group | Select-Object -Last 1).global_dispatch_id
        interpretation = $interpretation
    }) | Out-Null
}

$kernelExecutionUnitRows = New-Object System.Collections.Generic.List[object]
$kernelSummaryRows = New-Object System.Collections.Generic.List[object]
$allKernelNames = @((@($dispatchRowsWithKernel | Select-Object -ExpandProperty kernel_entry -Unique) + @($gtpinRows | Select-Object -ExpandProperty kernel -Unique)) | Sort-Object -Unique)

foreach ($kernelName in $allKernelNames) {
    $ovKernelRows = @($dispatchRowsWithKernel | Where-Object { $_.kernel_entry -eq $kernelName } | Sort-Object global_dispatch_id)
    $gtpinKernelRows = @($gtpinRows | Where-Object { $_.kernel -eq $kernelName } | Sort-Object dispatch_id)
    $matchedKernelRows = @($joinedRows | Where-Object { $_.kernel_entry -eq $kernelName -and $_.ov_match -eq "matched" } | Sort-Object dispatch_id)
    $steadyOvKernelRows = @($ovKernelRows | Where-Object { $_.net_id_num -ne 0 })

    $kernelUnits = @(
        $ovKernelRows |
        Group-Object execution_unit_key |
        Sort-Object { ($_.Group | Sort-Object global_dispatch_id | Select-Object -First 1).global_dispatch_id }
    )

    foreach ($unit in $kernelUnits) {
        $group = @($unit.Group | Sort-Object global_dispatch_id)
        $first = $group[0]
        $logicalIndexValue = ""
        if ($executionUnitOrder.ContainsKey($first.execution_unit_key)) {
            $logicalIndexValue = $executionUnitOrder[$first.execution_unit_key]
        }

        $kernelExecutionUnitRows.Add([pscustomobject]@{
            model_label = $ModelLabel
            kernel_entry = $kernelName
            execution_unit_key = $first.execution_unit_key
            net_id = $first.net_id
            iteration = $first.iteration
            is_internal_network_candidate = $first.is_internal_network_candidate
            logical_execution_index = $logicalIndexValue
            ov_dispatch_rows = $group.Count
            matched_gtpin_rows = @($matchedKernelRows | Where-Object { $_.execution_unit_key -eq $first.execution_unit_key }).Count
            primitive_ids = Join-UniqueValues @($group | Select-Object -ExpandProperty primitive_id)
            implementations = Join-UniqueValues @($group | Select-Object -ExpandProperty implementation)
        }) | Out-Null
    }

    $steadyNetIds = @($steadyOvKernelRows | Select-Object -ExpandProperty net_id_num -Unique | Sort-Object)
    $stableSteadyNetworks = New-Object System.Collections.Generic.List[string]
    $unstableSteadyNetworks = New-Object System.Collections.Generic.List[string]
    foreach ($steadyNetId in $steadyNetIds) {
        $rowsForNet = @($steadyOvKernelRows | Where-Object { $_.net_id_num -eq $steadyNetId })
        $countsPerUnit = @(
            $rowsForNet |
            Group-Object execution_unit_key |
            ForEach-Object { $_.Count }
        )

        if ($countsPerUnit.Count -gt 0 -and (@($countsPerUnit | Select-Object -Unique).Count -eq 1)) {
            [void]$stableSteadyNetworks.Add([string]$steadyNetId)
        } else {
            [void]$unstableSteadyNetworks.Add([string]$steadyNetId)
        }
    }

    $steadyUnitCount = @($steadyOvKernelRows | Group-Object execution_unit_key).Count
    $setupUnitCount = @($ovKernelRows | Where-Object { $_.net_id_num -eq 0 } | Group-Object execution_unit_key).Count
    $directCountMatch = ($ovKernelRows.Count -eq $matchedKernelRows.Count)

    $status = ""
    $reason = ""
    if ($ovKernelRows.Count -eq 0) {
        $status = "gtpin_only_kernel"
        $reason = "Kernel appears only in the GTPin raw dump."
    } elseif ($gtpinKernelRows.Count -eq 0) {
        $status = "ov_only_kernel"
        $reason = "Kernel appears only in the OpenVINO dispatch dump."
    } elseif (-not $directCountMatch) {
        $status = "dispatch_id_count_mismatch"
        $reason = "Direct dispatch-id join count does not match the OpenVINO kernel count."
    } elseif ($unstableSteadyNetworks.Count -gt 0) {
        $status = "non_uniform_within_network"
        $reason = "Kernel does not repeat uniformly inside steady repeated network(s): " + ($unstableSteadyNetworks -join ",")
    } elseif ($steadyUnitCount -eq 0 -and $setupUnitCount -gt 0) {
        $status = "setup_only_kernel"
        $reason = "Kernel appears only in the internal/setup network candidate."
    } else {
        $status = "robust_for_multi_inference"
        $reason = "Kernel matches by dispatch id and is stable within each repeated steady network."
    }

    $kernelSummaryRows.Add([pscustomobject]@{
        model_label = $ModelLabel
        kernel_entry = $kernelName
        ov_dispatch_rows_total = $ovKernelRows.Count
        gtpin_dispatch_rows_total = $gtpinKernelRows.Count
        direct_matched_dispatch_rows = $matchedKernelRows.Count
        setup_execution_unit_count = $setupUnitCount
        steady_execution_unit_count = $steadyUnitCount
        steady_network_ids = Join-UniqueValues @($steadyNetIds)
        stable_steady_network_ids = Join-UniqueValues @($stableSteadyNetworks)
        unstable_steady_network_ids = Join-UniqueValues @($unstableSteadyNetworks)
        direct_count_match = if ($directCountMatch) { "true" } else { "false" }
        partition_status = $status
        partition_reason = $reason
        primitive_ids = Join-UniqueValues @($ovKernelRows | Select-Object -ExpandProperty primitive_id)
        implementations = Join-UniqueValues @($ovKernelRows | Select-Object -ExpandProperty implementation)
    }) | Out-Null
}

$dispatchJoinCsv = Join-Path $OutputDir "dispatch_gtpin_multi_inference_join.csv"
$ovOnlyCsv = Join-Path $OutputDir "dispatches_missing_in_gtpin.csv"
$executionUnitCsv = Join-Path $OutputDir "execution_unit_summary.csv"
$networkSummaryCsv = Join-Path $OutputDir "network_summary.csv"
$kernelExecutionUnitCsv = Join-Path $OutputDir "kernel_execution_unit_breakdown.csv"
$kernelSummaryCsv = Join-Path $OutputDir "kernel_multi_inference_summary.csv"
$reportMd = Join-Path $OutputDir "multi_inference_robustness_report.md"

$joinedRows | Sort-Object dispatch_id | Export-Csv -NoTypeInformation -LiteralPath $dispatchJoinCsv
$ovOnlyRows | Sort-Object dispatch_id | Export-Csv -NoTypeInformation -LiteralPath $ovOnlyCsv
$executionUnitSummaryRows | Sort-Object first_dispatch_id | Export-Csv -NoTypeInformation -LiteralPath $executionUnitCsv
$networkSummaryRows | Sort-Object {[int64]$_.net_id} | Export-Csv -NoTypeInformation -LiteralPath $networkSummaryCsv
$kernelExecutionUnitRows | Sort-Object kernel_entry, logical_execution_index, {[int64]$_.net_id}, {[int64]$_.iteration} | Export-Csv -NoTypeInformation -LiteralPath $kernelExecutionUnitCsv
$kernelSummaryRows | Sort-Object kernel_entry | Export-Csv -NoTypeInformation -LiteralPath $kernelSummaryCsv

$matchedRows = @($joinedRows | Where-Object { $_.ov_match -eq "matched" })
$missingOvRows = @($joinedRows | Where-Object { $_.ov_match -ne "matched" })
$setupExecutionUnits = @($executionUnitSummaryRows | Where-Object { $_.is_internal_network_candidate -eq "true" })
$steadyExecutionUnitSummaryRows = @($executionUnitSummaryRows | Where-Object { $_.is_internal_network_candidate -ne "true" })
$kernelRobustRows = @($kernelSummaryRows | Where-Object { $_.partition_status -eq "robust_for_multi_inference" })
$kernelProblemRows = @($kernelSummaryRows | Where-Object { $_.partition_status -ne "robust_for_multi_inference" })
$setupOnlyKernelRows = @($kernelSummaryRows | Where-Object { $_.partition_status -eq "setup_only_kernel" })
$pointerCoverageMatches = @($matchedRows | Where-Object { $_.pointer_coverage_match -eq "true" })
$kernelNameMatches = @($matchedRows | Where-Object { $_.kernel_name_match -eq "true" })

$report = New-Object System.Collections.Generic.List[string]
$report.Add("# Multi-Inference Robustness Report")
$report.Add("")
$report.Add("Model label: " + '`' + $ModelLabel + '`')
$report.Add("")
$report.Add("---")
$report.Add("")
$report.Add("## Current Flow")
$report.Add("")
$report.Add("- Inputs used by this script are raw GTPin dispatch data and the OpenVINO dispatch dump only.")
$report.Add("- Direct correlation still uses `GTPin DispatchId <-> OpenVINO global_dispatch_id`.")
$report.Add("- High-level repetition no longer groups by `iteration` alone.")
$report.Add("- The execution unit is now `(net_id, iteration)`.")
$report.Add("- `net_id=0` is treated as an internal/setup network candidate and is separated from steady repeated execution.")
$report.Add("")
$report.Add("---")
$report.Add("")
$report.Add("## Coverage")
$report.Add("")
$report.Add("- OpenVINO dispatch rows with kernel metadata: $($dispatchRowsWithKernel.Count)")
$report.Add("- GTPin raw dispatch rows parsed: $($gtpinRows.Count)")
$report.Add("- Direct dispatch-id matches: $($matchedRows.Count)")
$report.Add("- GTPin rows missing OpenVINO partner: $($missingOvRows.Count)")
$report.Add("- OpenVINO rows missing GTPin partner: $($ovOnlyRows.Count)")
$report.Add("- Matched rows with kernel-name agreement: $($kernelNameMatches.Count) / $($matchedRows.Count)")
$report.Add("- Matched rows with pointer-coverage agreement: $($pointerCoverageMatches.Count) / $($matchedRows.Count)")
$report.Add("")
$report.Add("---")
$report.Add("")
$report.Add("## Network Summary")
$report.Add("")
foreach ($row in ($networkSummaryRows | Sort-Object {[int64]$_.net_id})) {
    $report.Add("- net_id=$($row.net_id): execution_units=$($row.execution_unit_count), rows_total=$($row.rows_total), rows_per_execution_unit=$($row.rows_per_execution_unit), interpretation=$($row.interpretation)")
}
$report.Add("")
$report.Add("---")
$report.Add("")
$report.Add("## Execution Units")
$report.Add("")
$report.Add("- Internal/setup execution units: $($setupExecutionUnits.Count)")
$report.Add("- Steady repeated execution units: $($steadyExecutionUnitSummaryRows.Count)")
$report.Add("")
foreach ($row in ($executionUnitSummaryRows | Sort-Object first_dispatch_id | Select-Object -First 12)) {
    $logicalLabel = if ([string]::IsNullOrWhiteSpace([string]$row.logical_execution_index)) { "setup" } else { "logical_exec=" + $row.logical_execution_index }
    $report.Add("- $($row.execution_unit_key): net_id=$($row.net_id), iteration=$($row.iteration), $logicalLabel, rows=$($row.ov_dispatch_rows), dispatch_id_range=$($row.first_dispatch_id)->$($row.last_dispatch_id)")
}
$report.Add("")
$report.Add("---")
$report.Add("")
$report.Add("## Kernel Robustness Verdict")
$report.Add("")
$report.Add("- Kernels robust for repeated multi-inference behavior: $($kernelRobustRows.Count)")
$report.Add("- Setup-only kernels: $($setupOnlyKernelRows.Count)")
$report.Add("- Kernels needing inspection: $($kernelProblemRows.Count)")
$report.Add("")
$report.Add("A kernel is marked `robust_for_multi_inference` only when it:")
$report.Add("- matches directly by dispatch id")
$report.Add("- is stable within each repeated steady network")
$report.Add("- is not only a setup/internal kernel artifact")
$report.Add("")
$report.Add("---")
$report.Add("")
$report.Add("## Example Problem Cases")
$report.Add("")
$examples = @($kernelProblemRows | Sort-Object kernel_entry | Select-Object -First 15)
if ($examples.Count -eq 0) {
    $report.Add("- None")
} else {
    foreach ($row in $examples) {
        $report.Add("- " + '`' + $row.kernel_entry + '`' + " -> " + $row.partition_status)
        $report.Add("  Reason: $($row.partition_reason)")
    }
}
$report.Add("")
$report.Add("---")
$report.Add("")
$report.Add("## Generated Files")
$report.Add("")
$report.Add('- `dispatch_gtpin_multi_inference_join.csv`')
$report.Add('- `dispatches_missing_in_gtpin.csv`')
$report.Add('- `execution_unit_summary.csv`')
$report.Add('- `network_summary.csv`')
$report.Add('- `kernel_execution_unit_breakdown.csv`')
$report.Add('- `kernel_multi_inference_summary.csv`')

Set-Content -LiteralPath $reportMd -Value $report -Encoding UTF8

Write-Output "Wrote:"
Write-Output "  $dispatchJoinCsv"
Write-Output "  $ovOnlyCsv"
Write-Output "  $executionUnitCsv"
Write-Output "  $networkSummaryCsv"
Write-Output "  $kernelExecutionUnitCsv"
Write-Output "  $kernelSummaryCsv"
Write-Output "  $reportMd"
