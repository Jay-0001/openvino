/*========================== begin_copyright_notice ============================
Copyright (C) 2026 Intel Corporation

SPDX-License-Identifier: MIT
============================= end_copyright_notice ===========================*/

/*!
 * @file Implementation skeleton of the memory_axis_kernel_correlation tool
 */

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "memory_axis_kernel_correlation.h"
#include "gen_send_decoder.h"

using namespace gtpin;
using namespace std;

/* ============================================================================================= */
// Configuration
/* ============================================================================================= */
Knob<int>         knobMemoryAxisNumThreadBuckets("num_thread_buckets", 32, "Number of thread buckets. Default - 32, zero - maximum thread buckets");
Knob<bool>        knobMemoryAxisPerTileProfiling("per_tile_profiling", false, "Enable per-tile (subdevice) profiling");
Knob<bool>        knobMemoryAxisSkipZeroResults("skip_zero_results", false, "Skip zero results in the memory_axis_kernel_correlation output");
Knob<bool>        knobMemoryAxisRequestFinalDispatch("memory_axis_request_final_dispatch", true, "Request FINAL_DISPATCH callbacks when supported");
Knob<bool>        knobMemoryAxisIncludeNonInput("memory_axis_include_non_input", false, "Include non-input memory arguments such as write-only or read-write pointers");
Knob<std::string> knobMemoryAxisTextOutput("memory_axis_text_output", "memory_axis_kernel_correlation.txt", "Readable text output file");

namespace
{
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

    if (knobMemoryAxisIncludeNonInput)
    {
        return true;
    }

    return arg.accessType.IsRead();
}

double SafeRatio(uint64_t numerator, uint64_t denominator)
{
    if (denominator == 0)
    {
        return 0.0;
    }
    return static_cast<double>(numerator) / static_cast<double>(denominator);
}
}

/* ============================================================================================= */
// MemoryAxisKernelCorrelation implementation
/* ============================================================================================= */
MemoryAxisKernelCorrelation* MemoryAxisKernelCorrelation::Instance()
{
    static MemoryAxisKernelCorrelation instance;
    return &instance;
}

void MemoryAxisKernelCorrelation::OnKernelBuild(IGtKernelInstrument& instrumentor)
{
    const IGtKernel&            kernel         = instrumentor.Kernel();
    const IGtCfg&               cfg            = instrumentor.Cfg();
    const IGtGenCoder&          coder          = instrumentor.Coder();
    const IGtGenArch&           genArch        = GTPin_GetCore()->GenArch();
    const IGtGenModel&          genModel       = kernel.GenModel();
    IGtProfileBufferAllocator&  allocator      = instrumentor.ProfileBufferAllocator();
    IGtVregFactory&             vregs          = coder.VregFactory();
    bool                        is64BitCounter = Use64BitCounters(coder);

    uint32_t numThreadBuckets = (knobMemoryAxisNumThreadBuckets == 0) ? genModel.MaxThreadBuckets() : knobMemoryAxisNumThreadBuckets;
    uint32_t numTiles         = (knobMemoryAxisPerTileProfiling && coder.IsTileIdSupported()) ? genArch.MaxTiles(kernel.GpuPlatform()) : 1;
    GtProfileArray profileArray(sizeof(MemoryAxisKernelRecord), numTiles, numThreadBuckets);
    profileArray.Allocate(allocator);

    _tmpReg32 = vregs.MakeScratch();
    _addrReg  = vregs.MakeMsgAddrScratch();
    _dataReg  = vregs.MakeMsgDataScratch(is64BitCounter ? VREG_TYPE_QWORD : VREG_TYPE_DWORD);

    for (auto bblPtr : cfg.Bbls())
    {
        for (auto insPtr : bblPtr->Instructions())
        {
            const IGtIns& ins = *insPtr;
            MemoryInstructionMetrics metrics = AnalyzeMemoryInstruction(ins);
            if (metrics.IsZero())
            {
                continue;
            }

            GtGenProcedure memCode;
            GenerateMemoryCounterCode(memCode, coder, profileArray, metrics);
            InstrumentInstruction(instrumentor, ins, GtIpoint::Before(), memCode);
        }
    }

    _kernels.emplace(kernel.Id(), MemoryAxisKernelProfile(kernel, profileArray));
}

void MemoryAxisKernelCorrelation::OnKernelRun(IGtKernelDispatch& dispatcher)
{
    const IGtKernel& kernel = dispatcher.Kernel();
    auto it = _kernels.find(kernel.Id());
    if (it == _kernels.end())
    {
        return;
    }

    if (dispatcher.ExecStage().IsDispatch())
    {
        bool isProfileEnabled = false;
        bool requestedFinalDispatch = false;
        bool finalDispatchSupported = false;

        GtKernelExecDesc execDesc;
        dispatcher.GetExecDescriptor(execDesc);
        if (kernel.IsInstrumented() && IsKernelExecProfileEnabled(execDesc, kernel.GpuPlatform(), kernel.Name().Get()))
        {
            isProfileEnabled = InitializeProfileBuffer(dispatcher, it->second);
            if (isProfileEnabled && knobMemoryAxisRequestFinalDispatch)
            {
                requestedFinalDispatch = true;
                finalDispatchSupported = dispatcher.ReportFinalDispatchStage();
            }
        }

        dispatcher.SetProfilingMode(isProfileEnabled);
        if (isProfileEnabled)
        {
            MemoryAxisDispatchSnapshot snapshot = CaptureSnapshot(dispatcher,
                                                                 _nextDispatchSequence++,
                                                                 requestedFinalDispatch,
                                                                 finalDispatchSupported);
            _dispatchStates[dispatcher.DispatchId()] = {snapshot};
        }
        return;
    }

    auto stateIt = _dispatchStates.find(dispatcher.DispatchId());
    if ((stateIt != _dispatchStates.end()) && dispatcher.ExecStage().IsFinalDispatch() && dispatcher.IsProfilingEnabled())
    {
        const MemoryAxisDispatchSnapshot& previous = stateIt->second.snapshot;
        stateIt->second.snapshot = CaptureSnapshot(dispatcher,
                                                   previous.dispatchSequence,
                                                   previous.requestedFinalDispatch,
                                                   previous.finalDispatchSupported);
    }
}

void MemoryAxisKernelCorrelation::OnKernelComplete(IGtKernelDispatch& dispatcher)
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
    MemoryAxisDispatchSnapshot snapshot;
    if (dispatchIt != _dispatchStates.end())
    {
        snapshot = dispatchIt->second.snapshot;
    }
    else
    {
        snapshot = CaptureSnapshot(dispatcher, _nextDispatchSequence++, false, false);
    }

    const IGtProfileBuffer* buffer = dispatcher.GetProfileBuffer();
    const GtProfileArray& profileArray = kernelIt->second.GetProfileArray();
    GTPIN_ASSERT(buffer);

    for (uint32_t tileId = 0; tileId < profileArray.NumRecords(); ++tileId)
    {
        MemoryAxisDispatchProfile& dispatchProfile = kernelIt->second.AddKernelDispatch(snapshot, tileId);
        for (uint32_t threadBucket = 0; threadBucket < profileArray.NumThreadBuckets(); ++threadBucket)
        {
            MemoryAxisKernelRecord record = {};
            if (!profileArray.Read(*buffer, &record, tileId, 1, threadBucket))
            {
                GTPIN_ERROR_MSG(string("MEMORY_AXIS_KERNEL_CORRELATION : ") + string(kernel.Name()) + " : Failed to read from memory buffer");
            }
            else
            {
                dispatchProfile.Accumulate(record);
            }
        }
    }

    _dispatchStates.erase(dispatcher.DispatchId());
}

void MemoryAxisKernelCorrelation::OnFini()
{
    MemoryAxisKernelCorrelation& me = *Instance();
    string filePath = JoinPath(string(GTPin_GetCore()->ProfileDir()), string(knobMemoryAxisTextOutput));

    ofstream fs(filePath);
    if (fs.is_open())
    {
        fs << me.ToText();
        fs.close();
    }
    else
    {
        GTPIN_WARNING("MEMORY_AXIS_KERNEL_CORRELATION : could not create file: " + filePath);
    }
}

string MemoryAxisKernelCorrelation::ToText() const
{
    ostringstream os;
    os << "### Memory-axis kernel correlation profile generated by GTPin ###" << endl << endl;
    os << "Legend:" << endl;
    os << "  Dynamic counters below are directly feasible from instruction inspection plus SEND decoding." << endl;
    os << "  Estimated bytes are instruction-centric estimates, not hardware-observed bandwidth." << endl;
    os << "  Primary metrics are condensed to total memory operations and total estimated bytes." << endl;
    os << "  Secondary metrics summarize access mix without inflating the top-level schema." << endl << endl;

    for (const auto& kernelEntry : _kernels)
    {
        const MemoryAxisKernelProfile& kernelProfile = kernelEntry.second;
        for (const auto& dp : kernelProfile.DispatchProfiles())
        {
            uint64_t totalMemoryOps = dp.memReads + dp.memWrites + dp.atomics;
            uint64_t totalEstimatedBytes = dp.estimatedReadBytes + dp.estimatedWriteBytes;
            if (knobMemoryAxisSkipZeroResults &&
                totalMemoryOps == 0 && totalEstimatedBytes == 0)
            {
                continue;
            }

            double writeDominance = SafeRatio(dp.memWrites + dp.atomics, totalMemoryOps);
            double estimatedBytesPerMemOp = SafeRatio(totalEstimatedBytes, totalMemoryOps);

            os << "--------------------------------------------------------------------------------" << endl;
            os << "DispatchId: " << dp.snapshot.dispatchId << endl;
            os << "Kernel: " << kernelProfile.Name() << endl;
            os << "TotalMemoryOps: " << totalMemoryOps
               << "  EstimatedTotalBytes: " << totalEstimatedBytes
               << endl;
            os << "WriteDominance: " << fixed << setprecision(2) << (100.0 * writeDominance) << "%"
               << "  EstimatedBytesPerMemOp: " << fixed << setprecision(2) << estimatedBytesPerMemOp
               << endl;
            os << "AccessBreakdown:"
               << " Reads=" << dp.memReads
               << " Writes=" << dp.memWrites
               << " Atomics=" << dp.atomics
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
    }

    return os.str();
}

bool MemoryAxisKernelCorrelation::Use64BitCounters(const IGtGenCoder& coder)
{
    return coder.InstructionFactory().CanAccessAtomically(GED_DATA_TYPE_uq);
}

MemoryInstructionMetrics MemoryAxisKernelCorrelation::AnalyzeMemoryInstruction(const IGtIns& ins)
{
    MemoryInstructionMetrics metrics;
    if (!ins.IsMemAccess() || ins.IsEot())
    {
        return metrics;
    }

    metrics.memReads = ins.IsMemRead() ? 1 : 0;
    metrics.memWrites = ins.IsMemWrite() ? 1 : 0;
    metrics.atomics = ins.IsAtomic() ? 1 : 0;

    // Keep the same conservative decoding strategy as funtime_memory.
    uint64_t bytes = 0;
    if (ins.IsSendMessage())
    {
        DcSendMsg msg = DcSendMsg::Decode(ins.GetGedIns());
        if (msg.IsValid())
        {
            uint32_t numAccesses = ins.NumAccesses();
            uint32_t dataSize = msg.ElementSize() * msg.NumElements();
            if (numAccesses != 0 && dataSize != 0)
            {
                bytes = uint64_t(dataSize) * uint64_t(numAccesses);
            }
        }
    }

    metrics.estimatedReadBytes = ins.IsMemRead() ? bytes : 0;
    metrics.estimatedWriteBytes = ins.IsMemWrite() ? bytes : 0;
    return metrics;
}

void MemoryAxisKernelCorrelation::ComputeRecordAddress(GtGenProcedure& proc,
                                                       const IGtGenCoder& coder,
                                                       const GtProfileArray& profileArray)
{
    IGtInsFactory& insF = coder.InstructionFactory();
    if (profileArray.NumRecords() > 1)
    {
        GtReg& offsetReg = _tmpReg32;
        coder.LoadTileId(proc, offsetReg);
        proc += insF.MakeMul(offsetReg, offsetReg, sizeof(MemoryAxisKernelRecord));
        profileArray.ComputeAddress(coder, proc, _addrReg, offsetReg);
    }
    else
    {
        profileArray.ComputeAddress(coder, proc, _addrReg);
    }
}

void MemoryAxisKernelCorrelation::EmitCounterAdd(GtGenProcedure& proc,
                                                 const IGtGenCoder& coder,
                                                 const GtProfileArray& profileArray,
                                                 int32_t& baseOffset,
                                                 int32_t fieldOffset,
                                                 uint64_t value)
{
    if (value == 0)
    {
        return;
    }

    IGtInsFactory& insF = coder.InstructionFactory();
    bool is64BitCounter = Use64BitCounters(coder);
    int32_t offset = fieldOffset - baseOffset;
    profileArray.ComputeRelAddress(coder, proc, _addrReg, _addrReg, offset);
    baseOffset += offset;

    GtReg dataRegL = {_dataReg, sizeof(uint32_t), 0};
    proc += insF.MakeMov(dataRegL, static_cast<uint32_t>(value & 0xffffffffu));
    if (is64BitCounter)
    {
        GtReg dataRegH = {_dataReg, sizeof(uint32_t), 1};
        proc += insF.MakeMov(dataRegH, static_cast<uint32_t>((value >> 32) & 0xffffffffu));
        proc += insF.MakeAtomicAdd(NullReg(), _addrReg, _dataReg, GED_DATA_TYPE_uq);
    }
    else
    {
        proc += insF.MakeAtomicAdd(NullReg(), _addrReg, _dataReg, GED_DATA_TYPE_ud);
    }
}

void MemoryAxisKernelCorrelation::GenerateMemoryCounterCode(GtGenProcedure& proc,
                                                            const IGtGenCoder& coder,
                                                            const GtProfileArray& profileArray,
                                                            const MemoryInstructionMetrics& metrics)
{
    ComputeRecordAddress(proc, coder, profileArray);

    int32_t base = 0;
    EmitCounterAdd(proc, coder, profileArray, base, offsetof(MemoryAxisKernelRecord, memReads), metrics.memReads);
    EmitCounterAdd(proc, coder, profileArray, base, offsetof(MemoryAxisKernelRecord, memWrites), metrics.memWrites);
    EmitCounterAdd(proc, coder, profileArray, base, offsetof(MemoryAxisKernelRecord, atomics), metrics.atomics);
    EmitCounterAdd(proc, coder, profileArray, base, offsetof(MemoryAxisKernelRecord, estimatedReadBytes), metrics.estimatedReadBytes);
    EmitCounterAdd(proc, coder, profileArray, base, offsetof(MemoryAxisKernelRecord, estimatedWriteBytes), metrics.estimatedWriteBytes);

    if (!proc.empty()) { proc.front()->AppendAnnotation(__func__); }
}

bool MemoryAxisKernelCorrelation::InitializeProfileBuffer(IGtKernelDispatch& dispatcher, const MemoryAxisKernelProfile& kernelProfile)
{
    IGtProfileBuffer* buffer = dispatcher.CreateProfileBuffer();
    GTPIN_ASSERT(buffer);
    return kernelProfile.GetProfileArray().Initialize(*buffer);
}

MemoryAxisDispatchSnapshot MemoryAxisKernelCorrelation::CaptureSnapshot(const IGtKernelDispatch& dispatcher,
                                                                       uint64_t dispatchSequence,
                                                                       bool requestedFinalDispatch,
                                                                       bool finalDispatchSupported) const
{
    MemoryAxisDispatchSnapshot snapshot;
    const IGtKernel& kernel = dispatcher.Kernel();

    snapshot.dispatchId = dispatcher.DispatchId();
    snapshot.dispatchSequence = dispatchSequence;
    snapshot.execStage = dispatcher.ExecStage();
    snapshot.requestedFinalDispatch = requestedFinalDispatch;
    snapshot.finalDispatchSupported = finalDispatchSupported;
    snapshot.lateDispatchData = dispatcher.ExecStage().IsFinalDispatch();

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

        MemoryAxisArgRecord record;
        record.argOrdinal       = argOrdinal;
        record.explicitArgIndex = uint32_t(arg.index);
        record.argRole          = AccessRole(arg.accessType);
        record.accessType       = arg.accessType.ToString();
        record.addrModel        = arg.addrModel.ToString();
        record.argSize          = arg.size;
        record.argOffset        = arg.offset;

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

const GtMemoryBuffer* MemoryAxisKernelCorrelation::FindContainingBuffer(const GtMemoryBufferConstSpan& buffers, uintptr_t ptr)
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
// MemoryAxisDispatchProfile implementation
/* ============================================================================================= */
MemoryAxisDispatchProfile::MemoryAxisDispatchProfile(const MemoryAxisDispatchSnapshot& snapshotIn, uint32_t tile) :
    snapshot(snapshotIn), tileId(tile) {}

void MemoryAxisDispatchProfile::Accumulate(const MemoryAxisKernelRecord& record)
{
    memReads += record.memReads;
    memWrites += record.memWrites;
    atomics += record.atomics;
    estimatedReadBytes += record.estimatedReadBytes;
    estimatedWriteBytes += record.estimatedWriteBytes;
}

/* ============================================================================================= */
// MemoryAxisKernelProfile implementation
/* ============================================================================================= */
MemoryAxisKernelProfile::MemoryAxisKernelProfile(const IGtKernel& kernel, const GtProfileArray& profileArray) :
    _kernelId(kernel.Id()),
    _name(GlueString(kernel.Name())),
    _profileArray(profileArray) {}

MemoryAxisDispatchProfile& MemoryAxisKernelProfile::AddKernelDispatch(const MemoryAxisDispatchSnapshot& snapshot, uint32_t tile)
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
    MemoryAxisKernelCorrelation::Instance()->Register();
    atexit(MemoryAxisKernelCorrelation::OnFini);
}
