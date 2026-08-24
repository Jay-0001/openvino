/*========================== begin_copyright_notice ============================
Copyright (C) 2026 Intel Corporation

SPDX-License-Identifier: MIT
============================= end_copyright_notice ===========================*/

/*!
 * @file Implementation of the raw_exec_kernel_correlation tool
 */

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "raw_exec_kernel_correlation.h"

using namespace gtpin;
using namespace std;

/* ============================================================================================= */
// Configuration
/* ============================================================================================= */
Knob<int>         knobRawExecNumThreadBuckets("num_thread_buckets", 32, "Number of thread buckets. Default - 32, zero - maximum thread buckets");
Knob<bool>        knobRawExecPerTileProfiling("per_tile_profiling", false, "Enable per-tile (subdevice) profiling");
Knob<bool>        knobRawExecSkipZeroResults("skip_zero_results", false, "Skip zero results in the raw_exec_kernel_correlation output");
Knob<bool>        knobRawExecRequestFinalDispatch("raw_exec_request_final_dispatch", true, "Request FINAL_DISPATCH callbacks when supported");
Knob<bool>        knobRawExecIncludeNonInput("raw_exec_include_non_input", false, "Include non-input memory arguments such as write-only or read-write pointers");
Knob<std::string> knobRawExecTextOutput("raw_exec_text_output", "raw_exec_kernel_correlation.txt", "Readable text output file");
Knob<std::string> knobRawExecDispatchTsvOutput("raw_exec_dispatch_tsv_output", "raw_exec_kernel_correlation_dispatch.tsv", "Per-dispatch TSV output file");
Knob<std::string> knobRawExecArgTsvOutput("raw_exec_arg_tsv_output", "raw_exec_kernel_correlation_args.tsv", "Per-memory-argument TSV output file");
Knob<std::string> knobRawExecKernelSummaryTsvOutput("raw_exec_kernel_summary_tsv_output", "raw_exec_kernel_correlation_kernel_summary.tsv", "Per-kernel summary TSV output file");

namespace
{
const char* StageName(GtKernelExecStage stage)
{
    if (stage.IsDispatch())      { return "DISPATCH"; }
    if (stage.IsFinalDispatch()) { return "FINAL_DISPATCH"; }
    if (stage.IsComplete())      { return "COMPLETE"; }
    return "NONE";
}

string SanitizeField(string value)
{
    replace(value.begin(), value.end(), '\t', ' ');
    replace(value.begin(), value.end(), '\r', ' ');
    replace(value.begin(), value.end(), '\n', ' ');
    return value;
}

string BytesToHex(ConstByteSpan bytes)
{
    if (bytes.empty())
    {
        return "EMPTY";
    }

    if ((bytes.data() == nullptr) && (bytes.size() != 0))
    {
        return string("NULL(") + ToString(bytes.size()) + "B)";
    }

    ostringstream os;
    os << "0x" << hex << uppercase << setfill('0');
    for (size_t idx = 0; idx < bytes.size(); ++idx)
    {
        os << setw(2) << uint32_t(bytes[idx]);
    }
    return os.str();
}

string PointerToHex(uintptr_t value)
{
    ostringstream os;
    os << "0x" << hex << uppercase << setw(sizeof(uintptr_t) * 2) << setfill('0') << uint64_t(value);
    return os.str();
}

string PointerSizedValueToHex(ConstByteSpan bytes)
{
    uintptr_t value = 0;
    if ((bytes.data() != nullptr) && (bytes.size() >= sizeof(uintptr_t)))
    {
        std::memcpy(&value, bytes.data(), sizeof(uintptr_t));
    }
    return PointerToHex(value);
}

string AccessRole(const GtAccessType& accessType)
{
    if (accessType.IsReadOnly())  { return "INPUT"; }
    if (accessType.IsWriteOnly()) { return "OUTPUT"; }
    if (accessType.IsReadWrite()) { return "INOUT"; }
    if (accessType.IsRead())      { return "INPUT"; }
    if (accessType.IsWrite())     { return "OUTPUT"; }
    return "UNKNOWN";
}

bool ShouldIncludeArg(const GtKernelArgument& arg)
{
    if (!arg.IsMemoryPointer() || !arg.IsExplicit())
    {
        return false;
    }

    if (knobRawExecIncludeNonInput)
    {
        return true;
    }

    return arg.accessType.IsRead();
}

struct KernelAggregate
{
    uint64_t dispatchCount = 0;
    uint64_t totalCycles   = 0;
    uint64_t totalFreq     = 0;
    uint64_t totalSkipped  = 0;
};

template <typename DISPATCH_VIEW>
bool NeedDistinctDispatchSequence(const vector<DISPATCH_VIEW>& ordered)
{
    for (const auto& item : ordered)
    {
        if (item.dispatchProfile->snapshot.dispatchSequence != item.dispatchProfile->snapshot.dispatchId)
        {
            return true;
        }
    }
    return false;
}
}

/* ============================================================================================= */
// RawExecKernelCorrelation implementation
/* ============================================================================================= */
RawExecKernelCorrelation* RawExecKernelCorrelation::Instance()
{
    static RawExecKernelCorrelation instance;
    return &instance;
}

void RawExecKernelCorrelation::OnKernelBuild(IGtKernelInstrument& instrumentor)
{
    const IGtKernel&            kernel         = instrumentor.Kernel();
    const IGtCfg&               cfg            = instrumentor.Cfg();
    const IGtGenCoder&          coder          = instrumentor.Coder();
    const IGtGenArch&           genArch        = GTPin_GetCore()->GenArch();
    const IGtGenModel&          genModel       = kernel.GenModel();
    IGtProfileBufferAllocator&  allocator      = instrumentor.ProfileBufferAllocator();
    IGtVregFactory&             vregs          = coder.VregFactory();
    bool                        is64BitCounter = Use64BitCounters(coder);

    uint32_t numThreadBuckets = (knobRawExecNumThreadBuckets == 0) ? genModel.MaxThreadBuckets() : knobRawExecNumThreadBuckets;
    uint32_t numTiles         = (knobRawExecPerTileProfiling && coder.IsTileIdSupported()) ? genArch.MaxTiles(kernel.GpuPlatform()) : 1;
    GtProfileArray profileArray(sizeof(RawExecKernelRecord), numTiles, numThreadBuckets);
    profileArray.Allocate(allocator);

    _timeReg  = vregs.Make(VREG_TYPE_DWORD);
    _tmpReg32 = vregs.MakeScratch();
    _addrReg  = vregs.MakeMsgAddrScratch();
    _dataReg  = vregs.MakeMsgDataScratch(is64BitCounter ? VREG_TYPE_QWORD : VREG_TYPE_DWORD);

    GtGenProcedure preCode;
    GeneratePreCode(preCode, coder);
    GtGenProcedure postCode;
    GeneratePostCode(postCode, coder, profileArray);

    instrumentor.InstrumentEntries(preCode);

    for (auto bblPtr : cfg.ExitBbls())
    {
        const IGtIns& ins = bblPtr->LastIns();
        GTPIN_ASSERT(ins.IsEot());
        GtGenProcedure fakeConsumers;
        coder.GenerateFakeSrcConsumers(fakeConsumers, ins);
        instrumentor.InstrumentInstruction(ins, GtIpoint::Before(), fakeConsumers);
        instrumentor.InstrumentInstruction(ins, GtIpoint::Before(), postCode);
    }

    _kernels.emplace(kernel.Id(), RawExecKernelProfile(kernel, profileArray));
}

void RawExecKernelCorrelation::OnKernelRun(IGtKernelDispatch& dispatcher)
{
    const IGtKernel& kernel = dispatcher.Kernel();
    auto it = _kernels.find(kernel.Id());
    if (it == _kernels.end())
    {
        return;
    }

    RawExecKernelProfile& kernelProfile = it->second;

    if (dispatcher.ExecStage().IsDispatch())
    {
        bool isProfileEnabled = false;
        bool requestedFinalDispatch = false;
        bool finalDispatchSupported = false;

        GtKernelExecDesc execDesc;
        dispatcher.GetExecDescriptor(execDesc);
        if (kernel.IsInstrumented() && IsKernelExecProfileEnabled(execDesc, kernel.GpuPlatform(), kernel.Name().Get()))
        {
            isProfileEnabled = InitializeProfileBuffer(dispatcher, kernelProfile);
            if (isProfileEnabled && knobRawExecRequestFinalDispatch)
            {
                requestedFinalDispatch = true;
                finalDispatchSupported = dispatcher.ReportFinalDispatchStage();
            }
        }

        dispatcher.SetProfilingMode(isProfileEnabled);
        if (isProfileEnabled)
        {
            RawExecDispatchSnapshot snapshot = CaptureSnapshot(dispatcher, _nextDispatchSequence++, requestedFinalDispatch, finalDispatchSupported);
            _dispatchStates[dispatcher.DispatchId()] = {snapshot};
        }
        return;
    }

    auto stateIt = _dispatchStates.find(dispatcher.DispatchId());
    if ((stateIt != _dispatchStates.end()) && dispatcher.ExecStage().IsFinalDispatch() && dispatcher.IsProfilingEnabled())
    {
        const RawExecDispatchSnapshot& previous = stateIt->second.snapshot;
        stateIt->second.snapshot = CaptureSnapshot(dispatcher,
                                                   previous.dispatchSequence,
                                                   previous.requestedFinalDispatch,
                                                   previous.finalDispatchSupported);
    }
}

void RawExecKernelCorrelation::OnKernelComplete(IGtKernelDispatch& dispatcher)
{
    const IGtKernel& kernel = dispatcher.Kernel();
    GtKernelExecDesc execDesc;
    dispatcher.GetExecDescriptor(execDesc);
    if (!dispatcher.IsProfilingEnabled() || !IsKernelExecProfileEnabled(execDesc, kernel.GpuPlatform(), kernel.Name().Get()))
    {
        return;
    }

    auto kernelIt = _kernels.find(kernel.Id());
    if (kernelIt == _kernels.end())
    {
        return;
    }

    auto dispatchIt = _dispatchStates.find(dispatcher.DispatchId());
    RawExecDispatchSnapshot snapshot;
    if (dispatchIt != _dispatchStates.end())
    {
        snapshot = dispatchIt->second.snapshot;
    }
    else
    {
        snapshot = CaptureSnapshot(dispatcher, _nextDispatchSequence++, false, false);
    }

    snapshot.execStage = GtKernelExecStage::COMPLETE;

    const IGtProfileBuffer* buffer = dispatcher.GetProfileBuffer();
    const GtProfileArray& profileArray = kernelIt->second.GetProfileArray();
    GTPIN_ASSERT(buffer);

    for (uint32_t tileId = 0; tileId < profileArray.NumRecords(); ++tileId)
    {
        RawExecDispatchProfile& dispatchProfile = kernelIt->second.AddKernelDispatch(snapshot, tileId);

        for (uint32_t threadBucket = 0; threadBucket < profileArray.NumThreadBuckets(); ++threadBucket)
        {
            RawExecKernelRecord record = {};
            if (!profileArray.Read(*buffer, &record, tileId, 1, threadBucket))
            {
                GTPIN_ERROR_MSG(string("RAW_EXEC_KERNEL_CORRELATION : ") + string(kernel.Name()) + " : Failed to read from memory buffer");
            }
            else
            {
                dispatchProfile.Accumulate(record);
            }
        }
    }

    _dispatchStates.erase(dispatcher.DispatchId());
}

void RawExecKernelCorrelation::OnFini()
{
    RawExecKernelCorrelation& me = *Instance();
    string profileDir = string(GTPin_GetCore()->ProfileDir());

    const struct OutputFile
    {
        string path;
        string contents;
        const char* warningTag;
    } outputs[] = {
        {JoinPath(profileDir, string(knobRawExecTextOutput)),             me.ToText(),              "RAW_EXEC_KERNEL_CORRELATION"},
        {JoinPath(profileDir, string(knobRawExecDispatchTsvOutput)),      me.ToDispatchTsv(),       "RAW_EXEC_KERNEL_CORRELATION"},
        {JoinPath(profileDir, string(knobRawExecArgTsvOutput)),           me.ToArgTsv(),            "RAW_EXEC_KERNEL_CORRELATION"},
        {JoinPath(profileDir, string(knobRawExecKernelSummaryTsvOutput)), me.ToKernelSummaryTsv(),  "RAW_EXEC_KERNEL_CORRELATION"},
    };

    for (const auto& output : outputs)
    {
        ofstream fs(output.path);
        if (fs.is_open())
        {
            fs << output.contents;
            fs.close();
        }
        else
        {
            GTPIN_WARNING(string(output.warningTag) + " : could not create file: " + output.path);
        }
    }
}

string RawExecKernelCorrelation::ToText() const
{
    struct DispatchView
    {
        const RawExecKernelProfile*   kernelProfile;
        const RawExecDispatchProfile* dispatchProfile;
    };

    vector<DispatchView> ordered;
    uint64_t totalCycles = 0;
    uint64_t totalFreq   = 0;
    for (const auto& kernelEntry : _kernels)
    {
        for (const auto& dp : kernelEntry.second.DispatchProfiles())
        {
            if (knobRawExecSkipZeroResults && dp.freq == 0)
            {
                continue;
            }

            ordered.push_back({&kernelEntry.second, &dp});
            totalCycles += dp.cycles;
            totalFreq += dp.freq;
        }
    }

    sort(ordered.begin(), ordered.end(),
         [](const DispatchView& lhs, const DispatchView& rhs)
         {
             if (lhs.dispatchProfile->snapshot.dispatchSequence != rhs.dispatchProfile->snapshot.dispatchSequence)
             {
                 return lhs.dispatchProfile->snapshot.dispatchSequence < rhs.dispatchProfile->snapshot.dispatchSequence;
             }
             return lhs.dispatchProfile->tileId < rhs.dispatchProfile->tileId;
         });

    bool showDispatchSequence = NeedDistinctDispatchSequence(ordered);

    ostringstream os;
    os << "### Raw execution kernel correlation profile generated by GTPin ###" << endl << endl;
    os << "Session summary:" << endl;
    os << "  ProfiledDispatches=" << ordered.size()
       << "  TotalExecutionCycles=" << totalCycles
       << "  TotalInvocations=" << totalFreq
       << endl;
    os << "Legend:" << endl;
    os << "  The text report focuses on actionable execution cost plus memory-argument mapping data." << endl;
    os << "  Cycle-based metrics are the current time-domain signal available from this tool." << endl;
    os << "  A wall-clock conversion is not printed here because the required timer-frequency information is not exposed by the current API path." << endl;
    os << "  For true inference percentages, aggregate dispatch rows per inference on the OpenVINO side." << endl << endl;

    if (ordered.empty())
    {
        os << "No profiled kernel dispatches were recorded." << endl;
        return os.str();
    }

    for (const auto& item : ordered)
    {
        const RawExecKernelProfile& kp = *item.kernelProfile;
        const RawExecDispatchProfile& dp = *item.dispatchProfile;
        uint64_t avgCycles = (dp.freq == 0) ? 0 : (dp.cycles / dp.freq);

        os << "--------------------------------------------------------------------------------" << endl;
        os << "DispatchId: " << dp.snapshot.dispatchId;
        if (showDispatchSequence)
        {
            os << "  DispatchSeq: " << dp.snapshot.dispatchSequence;
        }
        os << endl;
        os << "Kernel: " << kp.Name() << endl;
        os << "InvocationCount: " << dp.freq
           << "  TotalExecutionCycles: " << dp.cycles
           << "  AvgExecutionCyclesPerInvocation: " << avgCycles
           << endl;

        if (dp.snapshot.memoryArgs.empty())
        {
            os << "MemoryArgs: NONE" << endl;
            continue;
        }

        os << "MemoryArgs:" << endl;
        for (const auto& arg : dp.snapshot.memoryArgs)
        {
            os << "  ArgOrdinal=" << arg.argOrdinal
               << " ExplicitArgIndex=" << arg.explicitArgIndex
               << " Role=" << arg.argRole
               << " Access=" << arg.accessType
               << endl;
            os << "    AddrModel=" << arg.addrModel
               << " Size=" << arg.argSize
               << " PayloadOffset=" << arg.argOffset
               << endl;
            os << "    RawValue=" << arg.rawValueHex
               << " PointerValue=" << (arg.pointerSizedValue ? arg.pointerValueHex : string("NA"))
               << endl;
            if (arg.matchedAllocatedBuffer)
            {
                os << "    MatchedBufferBase=" << arg.matchedBufferBaseHex
                   << " MatchedBufferEnd=" << arg.matchedBufferEndHex
                   << " MatchedBufferSize=" << arg.matchedBufferSize
                   << " MatchedBufferAccess=" << arg.matchedBufferAccess
                   << " MatchedBufferOffset=" << arg.matchedBufferOffset
                   << endl;
            }
            else
            {
                os << "    MatchedBuffer=NONE" << endl;
            }
        }
    }

    return os.str();
}

string RawExecKernelCorrelation::ToDispatchTsv() const
{
    struct DispatchView
    {
        const RawExecKernelProfile*   kernelProfile;
        const RawExecDispatchProfile* dispatchProfile;
    };

    vector<DispatchView> ordered;
    uint64_t totalCycles = 0;
    for (const auto& kernelEntry : _kernels)
    {
        for (const auto& dp : kernelEntry.second.DispatchProfiles())
        {
            if (knobRawExecSkipZeroResults && dp.freq == 0)
            {
                continue;
            }

            ordered.push_back({&kernelEntry.second, &dp});
            totalCycles += dp.cycles;
        }
    }

    sort(ordered.begin(), ordered.end(),
         [](const DispatchView& lhs, const DispatchView& rhs)
         {
             if (lhs.dispatchProfile->snapshot.dispatchSequence != rhs.dispatchProfile->snapshot.dispatchSequence)
             {
                 return lhs.dispatchProfile->snapshot.dispatchSequence < rhs.dispatchProfile->snapshot.dispatchSequence;
             }
             return lhs.dispatchProfile->tileId < rhs.dispatchProfile->tileId;
         });

    ostringstream os;
    os << "dispatch_sequence"
       << '\t' << "dispatch_id"
       << '\t' << "kernel_id"
       << '\t' << "kernel_hash_id"
       << '\t' << "kernel_name"
       << '\t' << "kernel_unique_name"
       << '\t' << "kernel_extended_name"
       << '\t' << "platform"
       << '\t' << "arg_capture_stage"
       << '\t' << "requested_final_dispatch"
       << '\t' << "final_dispatch_supported"
       << '\t' << "late_dispatch_data"
       << '\t' << "tile_id"
       << '\t' << "exec_desc"
       << '\t' << "freq"
       << '\t' << "total_cycles"
       << '\t' << "avg_cycles"
       << '\t' << "skipped"
       << '\t' << "pct_profiled_cycles"
       << '\t' << "memory_arg_count"
       << '\n';

    for (const auto& item : ordered)
    {
        const RawExecKernelProfile& kp = *item.kernelProfile;
        const RawExecDispatchProfile& dp = *item.dispatchProfile;
        double pctCycles = (totalCycles == 0) ? 0.0 : (100.0 * static_cast<double>(dp.cycles) / static_cast<double>(totalCycles));
        uint64_t avgCycles = (dp.freq == 0) ? 0 : (dp.cycles / dp.freq);

        os << dp.snapshot.dispatchSequence
           << '\t' << dp.snapshot.dispatchId
           << '\t' << uint32_t(kp.KernelId())
           << '\t' << "0x" << hex << uppercase << kp.HashId() << dec << nouppercase
           << '\t' << SanitizeField(kp.Name())
           << '\t' << SanitizeField(kp.UniqueName())
           << '\t' << SanitizeField(kp.ExtendedName())
           << '\t' << kp.Platform().ToString()
           << '\t' << StageName(dp.snapshot.execStage)
           << '\t' << (dp.snapshot.requestedFinalDispatch ? 1 : 0)
           << '\t' << (dp.snapshot.finalDispatchSupported ? 1 : 0)
           << '\t' << (dp.snapshot.lateDispatchData ? 1 : 0)
           << '\t' << dp.tileId
           << '\t' << SanitizeField(dp.snapshot.kernelExecDesc.ToString(kp.Platform(), ExecDescAlignedFormat()))
           << '\t' << dp.freq
           << '\t' << dp.cycles
           << '\t' << avgCycles
           << '\t' << dp.skipped
           << '\t' << fixed << setprecision(6) << pctCycles
           << '\t' << dp.snapshot.memoryArgs.size()
           << '\n';
    }

    return os.str();
}

string RawExecKernelCorrelation::ToArgTsv() const
{
    struct DispatchView
    {
        const RawExecKernelProfile*   kernelProfile;
        const RawExecDispatchProfile* dispatchProfile;
    };

    vector<DispatchView> ordered;
    for (const auto& kernelEntry : _kernels)
    {
        for (const auto& dp : kernelEntry.second.DispatchProfiles())
        {
            if (knobRawExecSkipZeroResults && dp.freq == 0)
            {
                continue;
            }
            ordered.push_back({&kernelEntry.second, &dp});
        }
    }

    sort(ordered.begin(), ordered.end(),
         [](const DispatchView& lhs, const DispatchView& rhs)
         {
             if (lhs.dispatchProfile->snapshot.dispatchSequence != rhs.dispatchProfile->snapshot.dispatchSequence)
             {
                 return lhs.dispatchProfile->snapshot.dispatchSequence < rhs.dispatchProfile->snapshot.dispatchSequence;
             }
             return lhs.dispatchProfile->tileId < rhs.dispatchProfile->tileId;
         });

    ostringstream os;
    os << "dispatch_sequence"
       << '\t' << "dispatch_id"
       << '\t' << "kernel_id"
       << '\t' << "kernel_name"
       << '\t' << "tile_id"
       << '\t' << "arg_capture_stage"
       << '\t' << "arg_ordinal"
       << '\t' << "explicit_arg_index"
       << '\t' << "arg_type"
       << '\t' << "arg_role"
       << '\t' << "access_type"
       << '\t' << "addr_space"
       << '\t' << "addr_model"
       << '\t' << "arg_size"
       << '\t' << "arg_offset"
       << '\t' << "raw_value_hex"
       << '\t' << "pointer_sized_value"
       << '\t' << "pointer_value_hex"
       << '\t' << "matched_allocated_buffer"
       << '\t' << "matched_buffer_base_hex"
       << '\t' << "matched_buffer_end_hex"
       << '\t' << "matched_buffer_size"
       << '\t' << "matched_buffer_access"
       << '\t' << "matched_buffer_offset"
       << '\n';

    for (const auto& item : ordered)
    {
        const RawExecKernelProfile& kp = *item.kernelProfile;
        const RawExecDispatchProfile& dp = *item.dispatchProfile;
        for (const auto& arg : dp.snapshot.memoryArgs)
        {
            os << dp.snapshot.dispatchSequence
               << '\t' << dp.snapshot.dispatchId
               << '\t' << uint32_t(kp.KernelId())
               << '\t' << SanitizeField(kp.Name())
               << '\t' << dp.tileId
               << '\t' << StageName(dp.snapshot.execStage)
               << '\t' << arg.argOrdinal
               << '\t' << arg.explicitArgIndex
               << '\t' << SanitizeField(arg.argType)
               << '\t' << arg.argRole
               << '\t' << arg.accessType
               << '\t' << arg.addrSpace
               << '\t' << arg.addrModel
               << '\t' << arg.argSize
               << '\t' << arg.argOffset
               << '\t' << arg.rawValueHex
               << '\t' << (arg.pointerSizedValue ? 1 : 0)
               << '\t' << arg.pointerValueHex
               << '\t' << (arg.matchedAllocatedBuffer ? 1 : 0)
               << '\t' << arg.matchedBufferBaseHex
               << '\t' << arg.matchedBufferEndHex
               << '\t' << arg.matchedBufferSize
               << '\t' << arg.matchedBufferAccess
               << '\t' << arg.matchedBufferOffset
               << '\n';
        }
    }

    return os.str();
}

string RawExecKernelCorrelation::ToKernelSummaryTsv() const
{
    vector<const RawExecKernelProfile*> kernels;
    uint64_t totalCycles = 0;

    for (const auto& kernelEntry : _kernels)
    {
        kernels.push_back(&kernelEntry.second);
        for (const auto& dp : kernelEntry.second.DispatchProfiles())
        {
            if (knobRawExecSkipZeroResults && dp.freq == 0)
            {
                continue;
            }
            totalCycles += dp.cycles;
        }
    }

    sort(kernels.begin(), kernels.end(),
         [](const RawExecKernelProfile* lhs, const RawExecKernelProfile* rhs)
         {
             return lhs->Name() < rhs->Name();
         });

    ostringstream os;
    os << "kernel_id"
       << '\t' << "kernel_hash_id"
       << '\t' << "kernel_name"
       << '\t' << "kernel_unique_name"
       << '\t' << "kernel_extended_name"
       << '\t' << "platform"
       << '\t' << "dispatch_count"
       << '\t' << "total_cycles"
       << '\t' << "avg_cycles_per_dispatch"
       << '\t' << "total_freq"
       << '\t' << "avg_cycles_per_freq"
       << '\t' << "total_skipped"
       << '\t' << "pct_profiled_cycles"
       << '\n';

    for (const auto* kp : kernels)
    {
        KernelAggregate agg;
        for (const auto& dp : kp->DispatchProfiles())
        {
            if (knobRawExecSkipZeroResults && dp.freq == 0)
            {
                continue;
            }

            ++agg.dispatchCount;
            agg.totalCycles += dp.cycles;
            agg.totalFreq += dp.freq;
            agg.totalSkipped += dp.skipped;
        }

        double pctCycles = (totalCycles == 0) ? 0.0 : (100.0 * static_cast<double>(agg.totalCycles) / static_cast<double>(totalCycles));
        uint64_t avgCyclesPerDispatch = (agg.dispatchCount == 0) ? 0 : (agg.totalCycles / agg.dispatchCount);
        uint64_t avgCyclesPerFreq = (agg.totalFreq == 0) ? 0 : (agg.totalCycles / agg.totalFreq);

        os << uint32_t(kp->KernelId())
           << '\t' << "0x" << hex << uppercase << kp->HashId() << dec << nouppercase
           << '\t' << SanitizeField(kp->Name())
           << '\t' << SanitizeField(kp->UniqueName())
           << '\t' << SanitizeField(kp->ExtendedName())
           << '\t' << kp->Platform().ToString()
           << '\t' << agg.dispatchCount
           << '\t' << agg.totalCycles
           << '\t' << avgCyclesPerDispatch
           << '\t' << agg.totalFreq
           << '\t' << avgCyclesPerFreq
           << '\t' << agg.totalSkipped
           << '\t' << fixed << setprecision(6) << pctCycles
           << '\n';
    }

    return os.str();
}

bool RawExecKernelCorrelation::Use64BitCounters(const IGtGenCoder& coder)
{
    return coder.InstructionFactory().CanAccessAtomically(GED_DATA_TYPE_uq);
}

void RawExecKernelCorrelation::GeneratePreCode(GtGenProcedure& proc, const IGtGenCoder& coder)
{
    coder.StartTimer(proc, _timeReg);
    if (!proc.empty()) { proc.front()->AppendAnnotation(__func__); }
}

void RawExecKernelCorrelation::ComputeRecordAddress(GtGenProcedure& proc, const IGtGenCoder& coder, const GtProfileArray& profileArray)
{
    IGtInsFactory& insF = coder.InstructionFactory();
    if (profileArray.NumRecords() > 1)
    {
        GtReg& offsetReg = _tmpReg32;
        coder.LoadTileId(proc, offsetReg);
        proc += insF.MakeMul(offsetReg, offsetReg, sizeof(RawExecKernelRecord));
        profileArray.ComputeAddress(coder, proc, _addrReg, offsetReg);
    }
    else
    {
        profileArray.ComputeAddress(coder, proc, _addrReg);
    }
}

void RawExecKernelCorrelation::GeneratePostCode(GtGenProcedure& proc, const IGtGenCoder& coder, const GtProfileArray& profileArray)
{
    IGtInsFactory& insF = coder.InstructionFactory();
    bool is64BitCounter = Use64BitCounters(coder);
    GtReg flagReg = FlagReg(0);
    GtReg dataRegL = {_dataReg, sizeof(uint32_t), 0};

    coder.StopTimerExt(proc, _timeReg);
    ComputeRecordAddress(proc, coder, profileArray);

    int32_t base = 0;
    int32_t offset = offsetof(RawExecKernelRecord, cycles) - base;
    profileArray.ComputeRelAddress(coder, proc, _addrReg, _addrReg, offset);
    base += offset;
    proc += insF.MakeMov(dataRegL, _timeReg);
    if (is64BitCounter)
    {
        GtReg dataRegH = {_dataReg, sizeof(uint32_t), 1};
        proc += insF.MakeMov(dataRegH, 0);
        proc += insF.MakeAtomicAdd(NullReg(), _addrReg, _dataReg, GED_DATA_TYPE_uq);
    }
    else
    {
        proc += insF.MakeAtomicAdd(NullReg(), _addrReg, _dataReg, GED_DATA_TYPE_ud);
    }

    offset = offsetof(RawExecKernelRecord, freq) - base;
    profileArray.ComputeRelAddress(coder, proc, _addrReg, _addrReg, offset);
    base += offset;
    proc += insF.MakeAtomicInc(NullReg(), _addrReg, GED_DATA_TYPE_ud);

    offset = offsetof(RawExecKernelRecord, skipped) - base;
    profileArray.ComputeRelAddress(coder, proc, _addrReg, _addrReg, offset);
    proc += insF.MakeAtomicInc(NullReg(), _addrReg, GED_DATA_TYPE_ud).SetPredicate(flagReg);

    if (!proc.empty()) { proc.front()->AppendAnnotation(__func__); }
}

bool RawExecKernelCorrelation::InitializeProfileBuffer(IGtKernelDispatch& dispatcher, const RawExecKernelProfile& kernelProfile)
{
    IGtProfileBuffer* buffer = dispatcher.CreateProfileBuffer();
    GTPIN_ASSERT(buffer);
    return kernelProfile.GetProfileArray().Initialize(*buffer);
}

RawExecDispatchSnapshot RawExecKernelCorrelation::CaptureSnapshot(const IGtKernelDispatch& dispatcher,
                                                                  uint64_t dispatchSequence,
                                                                  bool requestedFinalDispatch,
                                                                  bool finalDispatchSupported) const
{
    RawExecDispatchSnapshot snapshot;
    const IGtKernel& kernel = dispatcher.Kernel();

    snapshot.dispatchId = dispatcher.DispatchId();
    snapshot.dispatchSequence = dispatchSequence;
    snapshot.execStage = dispatcher.ExecStage();
    snapshot.requestedFinalDispatch = requestedFinalDispatch;
    snapshot.finalDispatchSupported = finalDispatchSupported;
    snapshot.lateDispatchData = dispatcher.ExecStage().IsFinalDispatch();
    dispatcher.GetExecDescriptor(snapshot.kernelExecDesc);

    GtMemoryBufferConstSpan allocatedBuffers;
    if (!dispatcher.IsCompleted())
    {
        allocatedBuffers = dispatcher.GetAllocatedBuffers();
    }

    uint32_t argOrdinal = 0;
    for (const auto& arg : kernel.PayloadArguments())
    {
        if (!ShouldIncludeArg(arg))
        {
            ++argOrdinal;
            continue;
        }

        RawExecArgRecord record;
        record.argOrdinal       = argOrdinal;
        record.explicitArgIndex = uint32_t(arg.index);
        record.argType          = (arg.type != nullptr) ? arg.type : "";
        record.argSize          = arg.size;
        record.argOffset        = arg.offset;
        record.argRole          = AccessRole(arg.accessType);
        record.accessType       = arg.accessType.ToString();
        record.addrSpace        = arg.addrSpace.ToString();
        record.addrModel        = arg.addrModel.ToString();

        ConstByteSpan argValue = dispatcher.GetPayloadArgumentValue(arg.index);
        record.rawValueHex = BytesToHex(argValue);

        if (argValue.size() == sizeof(uintptr_t))
        {
            record.pointerSizedValue = true;
            record.pointerValueHex = PointerSizedValueToHex(argValue);

            uintptr_t ptrValue = 0;
            if (argValue.data() != nullptr)
            {
                std::memcpy(&ptrValue, argValue.data(), sizeof(uintptr_t));
            }

            const GtMemoryBuffer* matchedBuffer = FindContainingBuffer(allocatedBuffers, ptrValue);
            if (matchedBuffer != nullptr)
            {
                record.matchedAllocatedBuffer = true;
                record.matchedBufferBaseHex = PointerToHex(matchedBuffer->Base());
                record.matchedBufferEndHex = PointerToHex(matchedBuffer->End());
                record.matchedBufferSize = matchedBuffer->Size();
                record.matchedBufferAccess = matchedBuffer->Access().ToString();
                record.matchedBufferOffset = uint64_t(ptrValue - matchedBuffer->Base());
            }
        }

        snapshot.memoryArgs.push_back(record);
        ++argOrdinal;
    }

    return snapshot;
}

const GtMemoryBuffer* RawExecKernelCorrelation::FindContainingBuffer(const GtMemoryBufferConstSpan& buffers, uintptr_t ptr)
{
    for (const auto& buffer : buffers)
    {
        if (buffer.Range().Contains(ptr))
        {
            return &buffer;
        }
    }
    return nullptr;
}

/* ============================================================================================= */
// RawExecDispatchProfile implementation
/* ============================================================================================= */
RawExecDispatchProfile::RawExecDispatchProfile(const RawExecDispatchSnapshot& snapshotIn, uint32_t tile) :
    snapshot(snapshotIn), tileId(tile) {}

void RawExecDispatchProfile::Accumulate(const RawExecKernelRecord& record)
{
    cycles += record.cycles;
    freq += record.freq;
    skipped += record.skipped;
}

/* ============================================================================================= */
// RawExecKernelProfile implementation
/* ============================================================================================= */
RawExecKernelProfile::RawExecKernelProfile(const IGtKernel& kernel, const GtProfileArray& profileArray) :
    _kernelId(kernel.Id()),
    _name(GlueString(kernel.Name())),
    _uniqueName(GlueString(kernel.UniqueName())),
    _extendedName(ExtendedKernelName(kernel)),
    _platform(kernel.GpuPlatform()),
    _hashId(kernel.HashId()),
    _profileArray(profileArray) {}

RawExecDispatchProfile& RawExecKernelProfile::AddKernelDispatch(const RawExecDispatchSnapshot& snapshot, uint32_t tile)
{
    _dispatchProfiles.emplace_back(snapshot, tile);
    return _dispatchProfiles.back();
}

/* ============================================================================================= */
// GTPin_Entry
/* ============================================================================================= */
EXPORT_C_FUNC void GTPin_Entry(int argc, const char *argv[])
{
    ConfigureGTPin(argc, argv);
    RawExecKernelCorrelation::Instance()->Register();
    atexit(RawExecKernelCorrelation::OnFini);
}
