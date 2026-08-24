/*========================== begin_copyright_notice ============================
Copyright (C) 2026 Intel Corporation

SPDX-License-Identifier: MIT
============================= end_copyright_notice ===========================*/

/*!
 * @file Memory-axis kernel correlation tool skeleton definitions
 */

#ifndef MEMORY_AXIS_KERNEL_CORRELATION_H_
#define MEMORY_AXIS_KERNEL_CORRELATION_H_

#include <list>
#include <map>
#include <string>
#include <vector>

#include "gtpin_api.h"
#include "gtpin_tool_utils.h"

using namespace gtpin;

/* ============================================================================================= */
// Struct MemoryAxisKernelRecord
/* ============================================================================================= */
/*!
 * Core dynamic counters that are directly feasible from the current GTPin APIs
 * and the instruction decoding path already used by funtime_memory.
 *
 * Keep this record intentionally conservative for the first iteration.
 * Secondary metrics are synthesized during text formatting instead of
 * becoming first-class stored fields.
 */
struct MemoryAxisKernelRecord
{
    uint64_t memReads            = 0; ///< Dynamic count of memory-read instructions
    uint64_t memWrites           = 0; ///< Dynamic count of memory-write instructions
    uint64_t atomics             = 0; ///< Dynamic count of atomic memory instructions
    uint64_t estimatedReadBytes  = 0; ///< Estimated bytes referenced by memory reads
    uint64_t estimatedWriteBytes = 0; ///< Estimated bytes referenced by memory writes
};

/* ============================================================================================= */
// Struct MemoryAxisArgRecord
/* ============================================================================================= */
struct MemoryAxisArgRecord
{
    uint32_t    argOrdinal             = 0;
    uint32_t    explicitArgIndex       = 0;
    std::string argRole;
    std::string accessType;
    std::string addrModel;
    uint32_t    argSize                = 0;
    uint32_t    argOffset              = 0;
    std::string rawValueHex;
    bool        pointerSizedValue      = false;
    std::string pointerValueHex;
    bool        matchedAllocatedBuffer = false;
    std::string matchedBufferBaseHex;
    std::string matchedBufferEndHex;
    uint64_t    matchedBufferSize      = 0;
    std::string matchedBufferAccess;
    uint64_t    matchedBufferOffset    = 0;
};

/* ============================================================================================= */
// Struct MemoryAxisDispatchSnapshot
/* ============================================================================================= */
struct MemoryAxisDispatchSnapshot
{
    uint64_t                     dispatchId             = 0;
    uint64_t                     dispatchSequence       = 0;
    GtKernelExecStage            execStage;
    bool                         requestedFinalDispatch = false;
    bool                         finalDispatchSupported = false;
    bool                         lateDispatchData       = false;
    std::vector<MemoryAxisArgRecord> memoryArgs;
};

/* ============================================================================================= */
// Struct MemoryAxisDispatchProfile
/* ============================================================================================= */
struct MemoryAxisDispatchProfile
{
    explicit MemoryAxisDispatchProfile(const MemoryAxisDispatchSnapshot& snapshot, uint32_t tile = 0);
    void Accumulate(const MemoryAxisKernelRecord& record);

    MemoryAxisDispatchSnapshot snapshot;
    uint32_t                  tileId               = 0;
    uint64_t                  memReads             = 0;
    uint64_t                  memWrites            = 0;
    uint64_t                  atomics              = 0;
    uint64_t                  estimatedReadBytes   = 0;
    uint64_t                  estimatedWriteBytes  = 0;
};

/* ============================================================================================= */
// Struct MemoryInstructionMetrics
/* ============================================================================================= */
struct MemoryInstructionMetrics
{
    uint64_t memReads            = 0;
    uint64_t memWrites           = 0;
    uint64_t atomics             = 0;
    uint64_t estimatedReadBytes  = 0;
    uint64_t estimatedWriteBytes = 0;

    bool IsZero() const
    {
        return memReads == 0 && memWrites == 0 && atomics == 0 &&
               estimatedReadBytes == 0 && estimatedWriteBytes == 0;
    }
};

/* ============================================================================================= */
// Class MemoryAxisKernelProfile
/* ============================================================================================= */
class MemoryAxisKernelProfile
{
public:
    MemoryAxisKernelProfile(const IGtKernel& kernel, const GtProfileArray& profileArray);

    MemoryAxisDispatchProfile& AddKernelDispatch(const MemoryAxisDispatchSnapshot& snapshot, uint32_t tile = 0);

    GtKernelId  KernelId() const { return _kernelId; }
    const std::string& Name() const { return _name; }
    const GtProfileArray& GetProfileArray() const { return _profileArray; }
    const std::list<MemoryAxisDispatchProfile>& DispatchProfiles() const { return _dispatchProfiles; }

private:
    GtKernelId                         _kernelId;
    std::string                        _name;
    GtProfileArray                     _profileArray;
    std::list<MemoryAxisDispatchProfile> _dispatchProfiles;
};

/* ============================================================================================= */
// Class MemoryAxisKernelCorrelation
/* ============================================================================================= */
class MemoryAxisKernelCorrelation : public GtTool
{
public:
    const char* Name() const override { return "memory_axis_kernel_correlation"; }

    void OnKernelBuild(IGtKernelInstrument& instrumentor) override;
    void OnKernelRun(IGtKernelDispatch& dispatcher) override;
    void OnKernelComplete(IGtKernelDispatch& dispatcher) override;

public:
    static MemoryAxisKernelCorrelation* Instance();
    static void                         OnFini();
    std::string                         ToText() const;

private:
    struct DispatchState
    {
        MemoryAxisDispatchSnapshot snapshot;
    };

    MemoryAxisKernelCorrelation() = default;
    MemoryAxisKernelCorrelation(const MemoryAxisKernelCorrelation&) = delete;
    MemoryAxisKernelCorrelation& operator=(const MemoryAxisKernelCorrelation&) = delete;
    ~MemoryAxisKernelCorrelation() = default;

    void GenerateMemoryCounterCode(GtGenProcedure& proc,
                                   const IGtGenCoder& coder,
                                   const GtProfileArray& profileArray,
                                   const MemoryInstructionMetrics& metrics);
    void ComputeRecordAddress(GtGenProcedure& proc, const IGtGenCoder& coder, const GtProfileArray& profileArray);
    void EmitCounterAdd(GtGenProcedure& proc,
                        const IGtGenCoder& coder,
                        const GtProfileArray& profileArray,
                        int32_t& baseOffset,
                        int32_t fieldOffset,
                        uint64_t value);
    bool InitializeProfileBuffer(IGtKernelDispatch& dispatcher, const MemoryAxisKernelProfile& kernelProfile);
    MemoryAxisDispatchSnapshot CaptureSnapshot(const IGtKernelDispatch& dispatcher,
                                              uint64_t dispatchSequence,
                                              bool requestedFinalDispatch,
                                              bool finalDispatchSupported) const;

    static bool                     Use64BitCounters(const IGtGenCoder& coder);
    static MemoryInstructionMetrics AnalyzeMemoryInstruction(const IGtIns& ins);
    static const GtMemoryBuffer*    FindContainingBuffer(const GtMemoryBufferConstSpan& buffers, uintptr_t ptr);

private:
    std::map<GtKernelId, MemoryAxisKernelProfile> _kernels;
    std::map<uint64_t, DispatchState>             _dispatchStates;
    uint64_t                                      _nextDispatchSequence = 0;

    GtReg _addrReg;
    GtReg _dataReg;
    GtReg _tmpReg32;
};

#endif
