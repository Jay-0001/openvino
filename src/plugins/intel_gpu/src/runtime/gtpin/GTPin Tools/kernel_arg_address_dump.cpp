/*========================== begin_copyright_notice ============================
Copyright (C) 2026 Intel Corporation

SPDX-License-Identifier: MIT
============================= end_copyright_notice ===========================*/

/*!
 * @file Implementation of the kernel_arg_address_dump tool
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "gtpin_api.h"
#include "gtpin_tool_utils.h"

using namespace gtpin;
using namespace std;

/* ============================================================================================= */
// Configuration
/* ============================================================================================= */
Knob<bool>        knobKernelArgDumpRequestFinalDispatch("kernel_arg_dump_request_final_dispatch", true, "Request FINAL_DISPATCH callbacks when supported");
Knob<bool>        knobKernelArgDumpIncludeNonInput("kernel_arg_dump_include_non_input", false, "Include non-input memory arguments such as write-only or read-write pointers");
Knob<std::string> knobKernelArgDumpTextOutput("kernel_arg_dump_text_output", "kernel_arg_address_dump.txt", "Readable text output file");
Knob<std::string> knobKernelArgDumpTsvOutput("kernel_arg_dump_tsv_output", "kernel_arg_address_dump.tsv", "TSV output file");

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

    if (knobKernelArgDumpIncludeNonInput)
    {
        return true;
    }

    return arg.accessType.IsRead();
}
}

/* ============================================================================================= */
// Struct KernelArgAddressRecord
/* ============================================================================================= */
struct KernelArgAddressRecord
{
    uint64_t         dispatchId               = 0;
    GtKernelId       kernelId                 = 0;
    uint64_t         kernelHashId             = 0;
    string           kernelName;
    string           kernelUniqueName;
    string           kernelExtendedName;
    string           platform;
    string           execStage;
    string           execDesc;
    bool             requestedFinalDispatch   = false;
    bool             finalDispatchSupported   = false;
    uint32_t         argOrdinal               = 0;
    uint32_t         explicitArgIndex         = 0;
    string           argType;
    uint32_t         argSize                  = 0;
    uint32_t         argOffset                = 0;
    string           argRole;
    string           accessType;
    string           addrSpace;
    string           addrModel;
    string           rawValueHex;
    bool             pointerSizedValue        = false;
    string           pointerValueHex;
    bool             matchedAllocatedBuffer   = false;
    string           matchedBufferBaseHex;
    string           matchedBufferEndHex;
    uint64_t         matchedBufferSize        = 0;
    string           matchedBufferAccess;
    uint64_t         matchedBufferOffset      = 0;
    bool             lateDispatchData         = false;
};

/* ============================================================================================= */
// Class KernelArgAddressDump
/* ============================================================================================= */
class KernelArgAddressDump : public GtTool
{
public:
    const char* Name() const override { return "kernel_arg_address_dump"; }

    void OnKernelRun(IGtKernelDispatch& dispatcher) override;

public:
    static KernelArgAddressDump* Instance();
    static void OnFini();

private:
    struct DispatchState
    {
        bool requestedFinalDispatch = false;
        bool finalDispatchSupported = false;
    };

    KernelArgAddressDump() = default;
    KernelArgAddressDump(const KernelArgAddressDump&) = delete;
    KernelArgAddressDump& operator=(const KernelArgAddressDump&) = delete;
    ~KernelArgAddressDump() = default;

    void CaptureDispatch(const IGtKernelDispatch& dispatcher, bool requestedFinalDispatch, bool finalDispatchSupported);
    vector<KernelArgAddressRecord> BuildRecords(const IGtKernelDispatch& dispatcher,
                                                bool requestedFinalDispatch,
                                                bool finalDispatchSupported) const;
    static const GtMemoryBuffer* FindContainingBuffer(const GtMemoryBufferConstSpan& buffers, uintptr_t ptr);
    string ToText() const;
    string ToTsv() const;

private:
    vector<KernelArgAddressRecord> _records;
    map<uint64_t, DispatchState>   _dispatchStates;
};

/* ============================================================================================= */
// KernelArgAddressDump implementation
/* ============================================================================================= */
KernelArgAddressDump* KernelArgAddressDump::Instance()
{
    static KernelArgAddressDump instance;
    return &instance;
}

void KernelArgAddressDump::OnKernelRun(IGtKernelDispatch& dispatcher)
{
    bool requestedFinalDispatch = false;
    bool finalDispatchSupported = false;

    if (dispatcher.ExecStage().IsDispatch())
    {
        if (knobKernelArgDumpRequestFinalDispatch)
        {
            requestedFinalDispatch = true;
            finalDispatchSupported = dispatcher.ReportFinalDispatchStage();
        }

        _dispatchStates[dispatcher.DispatchId()] = {requestedFinalDispatch, finalDispatchSupported};
        CaptureDispatch(dispatcher, requestedFinalDispatch, finalDispatchSupported);
        return;
    }

    auto it = _dispatchStates.find(dispatcher.DispatchId());
    if (it != _dispatchStates.end())
    {
        requestedFinalDispatch = it->second.requestedFinalDispatch;
        finalDispatchSupported = it->second.finalDispatchSupported;
    }

    if (dispatcher.ExecStage().IsFinalDispatch())
    {
        CaptureDispatch(dispatcher, requestedFinalDispatch, finalDispatchSupported);
        _dispatchStates.erase(dispatcher.DispatchId());
    }
}

void KernelArgAddressDump::CaptureDispatch(const IGtKernelDispatch& dispatcher, bool requestedFinalDispatch, bool finalDispatchSupported)
{
    vector<KernelArgAddressRecord> newRecords = BuildRecords(dispatcher, requestedFinalDispatch, finalDispatchSupported);
    _records.insert(_records.end(), newRecords.begin(), newRecords.end());
}

vector<KernelArgAddressRecord> KernelArgAddressDump::BuildRecords(const IGtKernelDispatch& dispatcher,
                                                                  bool requestedFinalDispatch,
                                                                  bool finalDispatchSupported) const
{
    vector<KernelArgAddressRecord> records;
    const IGtKernel& kernel = dispatcher.Kernel();
    GtKernelExecDesc execDesc;
    dispatcher.GetExecDescriptor(execDesc);

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

        KernelArgAddressRecord record;
        record.dispatchId             = dispatcher.DispatchId();
        record.kernelId               = kernel.Id();
        record.kernelHashId           = kernel.HashId();
        record.kernelName             = SanitizeField(GlueString(kernel.Name()));
        record.kernelUniqueName       = SanitizeField(kernel.UniqueName());
        record.kernelExtendedName     = SanitizeField(ExtendedKernelName(kernel));
        record.platform               = kernel.GpuPlatform().ToString();
        record.execStage              = StageName(dispatcher.ExecStage());
        record.execDesc               = SanitizeField(execDesc.ToString(kernel.GpuPlatform(), ExecDescAlignedFormat()));
        record.requestedFinalDispatch = requestedFinalDispatch;
        record.finalDispatchSupported = finalDispatchSupported;
        record.argOrdinal             = argOrdinal;
        record.explicitArgIndex       = uint32_t(arg.index);
        record.argType                = (arg.type != nullptr) ? arg.type : "";
        record.argSize                = arg.size;
        record.argOffset              = arg.offset;
        record.argRole                = AccessRole(arg.accessType);
        record.accessType             = arg.accessType.ToString();
        record.addrSpace              = arg.addrSpace.ToString();
        record.addrModel              = arg.addrModel.ToString();
        record.lateDispatchData       = dispatcher.ExecStage().IsFinalDispatch();

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

        records.push_back(record);
        ++argOrdinal;
    }

    return records;
}

const GtMemoryBuffer* KernelArgAddressDump::FindContainingBuffer(const GtMemoryBufferConstSpan& buffers, uintptr_t ptr)
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

string KernelArgAddressDump::ToText() const
{
    ostringstream os;
    os << "### Kernel argument address dump generated by GTPin ###" << endl << endl;
    os << "Legend:" << endl;
    os << "Each entry corresponds to one explicit kernel payload argument that references memory." << endl;
    os << "By default the tool prints input-like arguments only; enable kernel_arg_dump_include_non_input to also include output and inout pointers." << endl;
    os << "pointer_value is the raw payload value interpreted as uintptr_t when the payload size matches pointer width." << endl;
    os << "matched_buffer_* fields come from IGtKernelDispatch::GetAllocatedBuffers and are most trustworthy on FINAL_DISPATCH when available." << endl;
    os << "For cl_mem-based flows, the payload identity may not be a device virtual address, so buffer matching can legitimately fail." << endl << endl;

    uint64_t currentDispatchId = uint64_t(-1);
    string currentStage;
    for (const auto& record : _records)
    {
        if ((record.dispatchId != currentDispatchId) || (record.execStage != currentStage))
        {
            currentDispatchId = record.dispatchId;
            currentStage = record.execStage;

            os << "--------------------------------------------------------------------------------" << endl;
            os << "DispatchId: " << record.dispatchId
               << "  Stage: " << record.execStage
               << "  RequestedFinalDispatch: " << (record.requestedFinalDispatch ? "Y" : "N")
               << "  FinalDispatchSupported: " << (record.finalDispatchSupported ? "Y" : "N")
               << endl;
            os << "Kernel: " << record.kernelName << endl;
            os << "UniqueName: " << record.kernelUniqueName << endl;
            os << "ExtendedName: " << record.kernelExtendedName << endl;
            os << "KernelHashId: 0x" << hex << uppercase << record.kernelHashId << dec << nouppercase << endl;
            os << "Platform: " << record.platform << endl;
            os << "ExecutionDescriptor: " << record.execDesc << endl;
        }

        os << endl;
        os << "ArgOrdinal=" << record.argOrdinal
           << "  ExplicitArgIndex=" << record.explicitArgIndex
           << "  Type=" << record.argType
           << "  Role=" << record.argRole
           << "  Access=" << record.accessType
           << endl;
        os << "  AddrSpace=" << record.addrSpace
           << "  AddrModel=" << record.addrModel
           << "  Size=" << record.argSize
           << "  PayloadOffset=" << record.argOffset
           << endl;
        os << "  RawValue=" << record.rawValueHex << endl;
        if (record.pointerSizedValue)
        {
            os << "  PointerValue=" << record.pointerValueHex << endl;
        }
        else
        {
            os << "  PointerValue=NA" << endl;
        }

        if (record.matchedAllocatedBuffer)
        {
            os << "  MatchedBufferBase=" << record.matchedBufferBaseHex
               << "  MatchedBufferEnd=" << record.matchedBufferEndHex
               << "  MatchedBufferSize=" << record.matchedBufferSize
               << "  MatchedBufferAccess=" << record.matchedBufferAccess
               << "  MatchedBufferOffset=" << record.matchedBufferOffset
               << endl;
        }
        else
        {
            os << "  MatchedBuffer=NONE" << endl;
        }
    }

    if (_records.empty())
    {
        os << "No matching kernel memory arguments were observed." << endl;
    }

    return os.str();
}

string KernelArgAddressDump::ToTsv() const
{
    ostringstream os;
    os << "dispatch_id"
       << '\t' << "stage"
       << '\t' << "requested_final_dispatch"
       << '\t' << "final_dispatch_supported"
       << '\t' << "late_dispatch_data"
       << '\t' << "kernel_id"
       << '\t' << "kernel_hash_id"
       << '\t' << "kernel_name"
       << '\t' << "kernel_unique_name"
       << '\t' << "kernel_extended_name"
       << '\t' << "platform"
       << '\t' << "exec_desc"
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

    for (const auto& record : _records)
    {
        os << record.dispatchId
           << '\t' << record.execStage
           << '\t' << (record.requestedFinalDispatch ? 1 : 0)
           << '\t' << (record.finalDispatchSupported ? 1 : 0)
           << '\t' << (record.lateDispatchData ? 1 : 0)
           << '\t' << uint32_t(record.kernelId)
           << '\t' << "0x" << hex << uppercase << record.kernelHashId << dec << nouppercase
           << '\t' << record.kernelName
           << '\t' << record.kernelUniqueName
           << '\t' << record.kernelExtendedName
           << '\t' << record.platform
           << '\t' << record.execDesc
           << '\t' << record.argOrdinal
           << '\t' << record.explicitArgIndex
           << '\t' << record.argType
           << '\t' << record.argRole
           << '\t' << record.accessType
           << '\t' << record.addrSpace
           << '\t' << record.addrModel
           << '\t' << record.argSize
           << '\t' << record.argOffset
           << '\t' << record.rawValueHex
           << '\t' << (record.pointerSizedValue ? 1 : 0)
           << '\t' << record.pointerValueHex
           << '\t' << (record.matchedAllocatedBuffer ? 1 : 0)
           << '\t' << record.matchedBufferBaseHex
           << '\t' << record.matchedBufferEndHex
           << '\t' << record.matchedBufferSize
           << '\t' << record.matchedBufferAccess
           << '\t' << record.matchedBufferOffset
           << '\n';
    }

    return os.str();
}

void KernelArgAddressDump::OnFini()
{
    KernelArgAddressDump& me = *Instance();
    string textPath = JoinPath(string(GTPin_GetCore()->ProfileDir()), string(knobKernelArgDumpTextOutput));
    string tsvPath  = JoinPath(string(GTPin_GetCore()->ProfileDir()), string(knobKernelArgDumpTsvOutput));

    ofstream textFs(textPath);
    if (textFs.is_open())
    {
        textFs << me.ToText();
        textFs.close();
    }
    else
    {
        GTPIN_WARNING("KERNEL_ARG_ADDRESS_DUMP : could not create file: " + textPath);
    }

    ofstream tsvFs(tsvPath);
    if (tsvFs.is_open())
    {
        tsvFs << me.ToTsv();
        tsvFs.close();
    }
    else
    {
        GTPIN_WARNING("KERNEL_ARG_ADDRESS_DUMP : could not create file: " + tsvPath);
    }
}

/* ============================================================================================= */
// GTPin_Entry
/* ============================================================================================= */
EXPORT_C_FUNC void GTPin_Entry(int argc, const char *argv[])
{
    ConfigureGTPin(argc, argv);
    KernelArgAddressDump::Instance()->Register();
    atexit(KernelArgAddressDump::OnFini);
}
