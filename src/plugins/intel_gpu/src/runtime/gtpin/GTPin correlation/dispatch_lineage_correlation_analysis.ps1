param(
    [Parameter(Mandatory = $true)]
    [string]$DispatchCsv,

    [Parameter(Mandatory = $true)]
    [string]$GtpinDump,

    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path (Split-Path -Parent $DispatchCsv) "dispatch_lineage_analysis"
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
            [void]$set.Add($identity)
        }
    }
    return $set
}

function Get-GtpinPointerList {
    param([string]$Value)
    if ([string]::IsNullOrWhiteSpace($Value)) {
        return @()
    }

    return @($Value -split ";" |
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

function Parse-GtpinDispatchDump {
    param([string]$Path)

    $dispatches = New-Object System.Collections.Generic.List[object]
    $current = $null

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
                pointer_values = New-Object System.Collections.Generic.List[string]
                arg_ordinals = New-Object System.Collections.Generic.List[string]
                raw_pointer_records = New-Object System.Collections.Generic.List[string]
            }
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

        if ($line -match '^ArgOrdinal=(\d+).*Type=arg_bypointer') {
            $current.arg_ordinals.Add($matches[1])
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

    $inputAddressSet = Get-AddressSet $_.input_arg_addresses
    $outputAddressSet = Get-AddressSet $_.output_memory_addresses

    [pscustomobject]@{
        net_id = $_.net_id
        iteration = $_.iteration
        dispatch_index = $_.dispatch_index
        global_dispatch_id = $globalDispatchId
        primitive_id = $_.primitive_id
        primitive_type = $_.primitive_type
        implementation = $_.implementation
        kernel_index = $_.kernel_index
        kernel_entry = $_.kernel_entry
        batch_hash = $_.batch_hash
        input_arg_addresses = $_.input_arg_addresses
        output_arg_addresses = if ($_.PSObject.Properties.Name -contains "output_arg_addresses") { $_.output_arg_addresses } else { "" }
        output_memory_addresses = if ($_.PSObject.Properties.Name -contains "output_memory_addresses") { $_.output_memory_addresses } else { "" }
        input_address_set = $inputAddressSet
        output_address_set = $outputAddressSet
    }
} | Sort-Object global_dispatch_id

$gtpinRows = Parse-GtpinDispatchDump -Path $GtpinDump | Sort-Object dispatch_id

$ovByGlobalId = @{}
foreach ($row in $dispatchRows) {
    $ovByGlobalId[$row.global_dispatch_id] = $row
}

$joinedRows = New-Object System.Collections.Generic.List[object]
foreach ($gtpinRow in $gtpinRows) {
    $ovRow = $null
    if ($ovByGlobalId.ContainsKey($gtpinRow.dispatch_id)) {
        $ovRow = $ovByGlobalId[$gtpinRow.dispatch_id]
    }

    $ovInputPointers = @()
    $ovOutputPointers = @()
    $ovCombinedPointers = @()
    $gtpinPointerList = @()
    $gtpinPointerSet = [System.Collections.Generic.HashSet[string]]::new()
    $inputPointersCovered = $false
    $outputPointersCovered = $false
    $inputOutputPointersCovered = $false
    $pointerSetMatch = $false
    $pointerOrderExactMatch = $false

    if ($null -ne $ovRow) {
        $ovInputPointers = @(Get-AddressEntries $ovRow.input_arg_addresses | ForEach-Object { Get-AddressIdentity $_ } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | ForEach-Object { $_.ToUpper() })
        $ovOutputPointers = @(Get-AddressEntries $ovRow.output_arg_addresses | ForEach-Object { Get-AddressIdentity $_ } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | ForEach-Object { $_.ToUpper() })
        $ovCombinedPointers = @($ovInputPointers + $ovOutputPointers)
        $gtpinPointerList = @(Get-GtpinPointerList ($gtpinRow.pointer_values -join ";"))
        $gtpinPointerSet = [System.Collections.Generic.HashSet[string]]::new([string[]]$gtpinPointerList)
        $inputPointersCovered = Test-PointerCoverage -Expected $ovInputPointers -ActualSet $gtpinPointerSet
        $outputPointersCovered = Test-PointerCoverage -Expected $ovOutputPointers -ActualSet $gtpinPointerSet
        $inputOutputPointersCovered = $inputPointersCovered -and $outputPointersCovered
        $pointerSetMatch = ([System.Collections.Generic.HashSet[string]]::new([string[]]$ovCombinedPointers)).SetEquals($gtpinPointerSet)
        $pointerOrderExactMatch = (($ovCombinedPointers -join ";") -eq ($gtpinPointerList -join ";"))
    }

    $joinedRows.Add([pscustomobject]@{
        global_dispatch_id = $gtpinRow.dispatch_id
        ov_match = if ($null -ne $ovRow) { "matched" } else { "missing_ov_dispatch" }
        primitive_id = if ($null -ne $ovRow) { $ovRow.primitive_id } else { "" }
        primitive_type = if ($null -ne $ovRow) { $ovRow.primitive_type } else { "" }
        implementation = if ($null -ne $ovRow) { $ovRow.implementation } else { "" }
        kernel_entry = if ($null -ne $ovRow) { $ovRow.kernel_entry } else { "" }
        net_id = if ($null -ne $ovRow) { $ovRow.net_id } else { "" }
        iteration = if ($null -ne $ovRow) { $ovRow.iteration } else { "" }
        dispatch_index = if ($null -ne $ovRow) { $ovRow.dispatch_index } else { "" }
        ov_input_arg_addresses = if ($null -ne $ovRow) { $ovRow.input_arg_addresses } else { "" }
        ov_output_arg_addresses = if ($null -ne $ovRow) { $ovRow.output_arg_addresses } else { "" }
        ov_output_memory_addresses = if ($null -ne $ovRow) { $ovRow.output_memory_addresses } else { "" }
        gtpin_kernel = $gtpinRow.kernel
        gtpin_unique_name = $gtpinRow.unique_name
        gtpin_extended_name = $gtpinRow.extended_name
        gtpin_arg_ordinals = ($gtpinRow.arg_ordinals -join ";")
        gtpin_pointer_values = ($gtpinRow.pointer_values -join ";")
        gtpin_raw_pointer_records = ($gtpinRow.raw_pointer_records -join ";")
        ov_gtpin_input_pointers_covered = if ($inputPointersCovered) { "true" } else { "false" }
        ov_gtpin_output_pointers_covered = if ($outputPointersCovered) { "true" } else { "false" }
        ov_gtpin_input_output_pointers_covered = if ($inputOutputPointersCovered) { "true" } else { "false" }
        ov_gtpin_pointer_set_match = if ($pointerSetMatch) { "true" } else { "false" }
        ov_gtpin_pointer_order_exact_match = if ($pointerOrderExactMatch) { "true" } else { "false" }
    })
}

$lineageEdges = New-Object System.Collections.Generic.List[object]
for ($i = 0; $i -lt $dispatchRows.Count; ++$i) {
    $producer = $dispatchRows[$i]
    if ($producer.output_address_set.Count -eq 0) {
        continue
    }

    for ($j = $i + 1; $j -lt $dispatchRows.Count; ++$j) {
        $consumer = $dispatchRows[$j]
        if ($consumer.input_address_set.Count -eq 0) {
            continue
        }

        $sharedAddresses = New-Object System.Collections.Generic.List[string]
        foreach ($address in $producer.output_address_set) {
            if ($consumer.input_address_set.Contains($address)) {
                $sharedAddresses.Add($address)
            }
        }

        if ($sharedAddresses.Count -eq 0) {
            continue
        }

        $lineageEdges.Add([pscustomobject]@{
            producer_global_dispatch_id = $producer.global_dispatch_id
            producer_primitive_id = $producer.primitive_id
            producer_kernel_entry = $producer.kernel_entry
            consumer_global_dispatch_id = $consumer.global_dispatch_id
            consumer_primitive_id = $consumer.primitive_id
            consumer_kernel_entry = $consumer.kernel_entry
            dispatch_distance = ($consumer.global_dispatch_id - $producer.global_dispatch_id)
            shared_buffer_count = $sharedAddresses.Count
            shared_buffer_addresses = ($sharedAddresses -join ";")
        })
    }
}

$nearestConsumerByProducer = $lineageEdges |
    Group-Object producer_global_dispatch_id |
    ForEach-Object {
        $_.Group | Sort-Object dispatch_distance, consumer_global_dispatch_id | Select-Object -First 1
    }

$joinedPath = Join-Path $OutputDir "dispatch_gtpin_join.csv"
$lineagePath = Join-Path $OutputDir "buffer_lineage_edges.csv"
$nearestPath = Join-Path $OutputDir "buffer_lineage_nearest_consumers.csv"
$summaryPath = Join-Path $OutputDir "dispatch_lineage_summary.md"

$joinedRows | Sort-Object global_dispatch_id | Export-Csv -NoTypeInformation -LiteralPath $joinedPath
$lineageEdges | Sort-Object producer_global_dispatch_id, consumer_global_dispatch_id | Export-Csv -NoTypeInformation -LiteralPath $lineagePath
$nearestConsumerByProducer | Sort-Object producer_global_dispatch_id | Export-Csv -NoTypeInformation -LiteralPath $nearestPath

$matchedCount = @($joinedRows | Where-Object { $_.ov_match -eq "matched" }).Count
$missingCount = $joinedRows.Count - $matchedCount
$pointerCoveredCount = @($joinedRows | Where-Object { $_.ov_gtpin_input_output_pointers_covered -eq "true" }).Count
$pointerSetMatchCount = @($joinedRows | Where-Object { $_.ov_gtpin_pointer_set_match -eq "true" }).Count
$pointerOrderMatchCount = @($joinedRows | Where-Object { $_.ov_gtpin_pointer_order_exact_match -eq "true" }).Count
$producerCount = @($dispatchRows | Where-Object { $_.output_address_set.Count -gt 0 }).Count
$consumerCount = @($dispatchRows | Where-Object { $_.input_address_set.Count -gt 0 }).Count
$lineageProducerCount = @($lineageEdges | Select-Object -ExpandProperty producer_global_dispatch_id -Unique).Count

$report = New-Object System.Collections.Generic.List[string]
$report.Add("# Dispatch Lineage Correlation Summary")
$report.Add("")
$report.Add("## Coverage")
$report.Add("")
$report.Add("- OV dispatch rows: $($dispatchRows.Count)")
$report.Add("- GTPin dispatch rows: $($gtpinRows.Count)")
$report.Add("- Matched by global dispatch id: $matchedCount")
$report.Add("- Missing OV matches from GTPin ids: $missingCount")
$report.Add("- Rows with OV input+output pointers covered by GTPin pointers: $pointerCoveredCount")
$report.Add("- Rows with OV/GTPin pointer set equality: $pointerSetMatchCount")
$report.Add("- Rows with exact OV/GTPin pointer ordering equality: $pointerOrderMatchCount")
$report.Add("")
$report.Add("## Lineage")
$report.Add("")
$report.Add("- Dispatches with output buffers: $producerCount")
$report.Add("- Dispatches with input buffers: $consumerCount")
$report.Add("- Producer dispatches with at least one downstream consumer: $lineageProducerCount")
$report.Add("- Total producer-consumer edges: $($lineageEdges.Count)")
$report.Add("")
$report.Add("## Interpretation")
$report.Add("")
$report.Add("- `dispatch_gtpin_join.csv` is the primary dispatch-to-dispatch alignment table.")
$report.Add("- `buffer_lineage_edges.csv` contains every future consumer that reuses an OV-recorded output buffer address.")
$report.Add("- `buffer_lineage_nearest_consumers.csv` keeps the shortest-distance consumer per producer as a compact lineage proxy.")
$report.Add("- This lineage is address-based, so buffer reuse can still create ambiguous long-range edges; the nearest-consumer view is usually the best first inspection layer.")

Set-Content -LiteralPath $summaryPath -Value $report -Encoding UTF8

Write-Output "Wrote:"
Write-Output "  $joinedPath"
Write-Output "  $lineagePath"
Write-Output "  $nearestPath"
Write-Output "  $summaryPath"
