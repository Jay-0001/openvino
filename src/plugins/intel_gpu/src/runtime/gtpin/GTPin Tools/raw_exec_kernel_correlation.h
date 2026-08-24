/*========================== begin_copyright_notice ============================
Copyright (C) 2026 Intel Corporation

SPDX-License-Identifier: MIT
============================= end_copyright_notice ===========================*/

/*!
 * @file Raw execution and memory-argument correlation profiling tool definitions
 */

#ifndef RAW_EXEC_KERNEL_CORRELATION_H_
#define RAW_EXEC_KERNEL_CORRELATION_H_

#include <list>
#include <map>
#include <string>
#include <vector>

#include "gtpin_api.h"
#include "gtpin_tool_utils.h"

using namespace gtpin;

/* ============================================================================================= */
// Struct RawExecKernelRecord
/* ============================================================================================= */
struct RawExecKernelRecord
{
    uint64_t cycles;
    uint32_t freq;
    uint32_t skipped;
};

/* ============================================================================================= */
// Struct RawExecArgRecord
/* ============================================================================================= */
struct RawExecArgRecord
{
    uint32_t    argOrdinal             = 0;
    uint32_t    explicitArgIndex       = 0;
    std::string argType;
    uint32_t    argSize                = 0;
    uint32_t    argOffset              = 0;
    std::string argRole;
    std::string accessType;
    std::string addrSpace;
    std::string addrModel;
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
// Struct RawExecDispatchSnapshot
/* ============================================================================================= */
struct RawExecDispatchSnapshot
{
    uint64_t                  dispatchId               = 0;
    uint64_t                  dispatchSequence         = 0;
    GtKernelExecDesc          kernelExecDesc;
    GtKernelExecStage         execStage;
    bool                      requestedFinalDispatch   = false;
    bool                      finalDispatchSupported   = false;
    bool                      lateDispatchData         = false;
    std::vector<RawExecArgRecord> memoryArgs;
};

/* ============================================================================================= */
// Struct RawExecDispatchProfile
/* ============================================================================================= */
struct RawExecDispatchProfile
{
    explicit RawExecDispatchProfile(const RawExecDispatchSnapshot& snapshot, uint32_t tile = 0);
    void Accumulate(const RawExecKernelRecord& record);

    RawExecDispatchSnapshot snapshot;
    uint32_t               tileId   = 0;
    uint64_t               cycles   = 0;
    uint64_t               freq     = 0;
    uint64_t               skipped  = 0;
};

/* ============================================================================================= */
// Class RawExecKernelProfile
/* ============================================================================================= */
class RawExecKernelProfile
{
public:
    RawExecKernelProfile(const IGtKernel& kernel, const GtProfileArray& profileArray);

    RawExecDispatchProfile& AddKernelDispatch(const RawExecDispatchSnapshot& snapshot, uint32_t tile = 0);

    GtKernelId                    KernelId()          const { return _kernelId; }
    const std::string&            Name()              const { return _name; }
    const std::string&            UniqueName()        const { return _uniqueName; }
    const std::string&            ExtendedName()      const { return _extendedName; }
    GtGpuPlatform                 Platform()          const { return _platform; }
    uint64_t                      HashId()            const { return _hashId; }
    const GtProfileArray&         GetProfileArray()   const { return _profileArray; }
    const std::list<RawExecDispatchProfile>& DispatchProfiles() const { return _dispatchProfiles; }

private:
    GtKernelId                    _kernelId;
    std::string                   _name;
    std::string                   _uniqueName;
    std::string                   _extendedName;
    GtGpuPlatform                 _platform;
    uint64_t                      _hashId;
    GtProfileArray                _profileArray;
    std::list<RawExecDispatchProfile> _dispatchProfiles;
};

/* ============================================================================================= */
// Class RawExecKernelCorrelation
/* ============================================================================================= */
class RawExecKernelCorrelation : public GtTool
{
public:
    const char* Name() const override { return "raw_exec_kernel_correlation"; }

    void OnKernelBuild(IGtKernelInstrument& instrumentor) override;
    void OnKernelRun(IGtKernelDispatch& dispatcher) override;
    void OnKernelComplete(IGtKernelDispatch& dispatcher) override;

public:
    static void                        OnFini();
    static RawExecKernelCorrelation*   Instance();
    std::string                        ToText() const;
    std::string                        ToDispatchTsv() const;
    std::string                        ToArgTsv() const;
    std::string                        ToKernelSummaryTsv() const;

private:
    struct DispatchState
    {
        RawExecDispatchSnapshot snapshot;
    };

    RawExecKernelCorrelation() = default;
    RawExecKernelCorrelation(const RawExecKernelCorrelation&) = delete;
    RawExecKernelCorrelation& operator=(const RawExecKernelCorrelation&) = delete;
    ~RawExecKernelCorrelation() = default;

    void GeneratePreCode(GtGenProcedure& proc, const IGtGenCoder& coder);
    void GeneratePostCode(GtGenProcedure& proc, const IGtGenCoder& coder, const GtProfileArray& profileArray);
    void ComputeRecordAddress(GtGenProcedure& proc, const IGtGenCoder& coder, const GtProfileArray& profileArray);
    bool InitializeProfileBuffer(IGtKernelDispatch& dispatcher, const RawExecKernelProfile& kernelProfile);
    RawExecDispatchSnapshot CaptureSnapshot(const IGtKernelDispatch& dispatcher,
                                            uint64_t dispatchSequence,
                                            bool requestedFinalDispatch,
                                            bool finalDispatchSupported) const;

    static bool                     Use64BitCounters(const IGtGenCoder& coder);
    static const GtMemoryBuffer*    FindContainingBuffer(const GtMemoryBufferConstSpan& buffers, uintptr_t ptr);

private:
    std::map<GtKernelId, RawExecKernelProfile> _kernels;
    std::map<uint64_t, DispatchState>          _dispatchStates;
    uint64_t                                   _nextDispatchSequence = 0;

    GtReg _addrReg;
    GtReg _dataReg;
    GtReg _timeReg;
    GtReg _tmpReg32;
};

#endif
