//temporary version -- not original
// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/event.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "intel_gpu/runtime/device_query.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include "ocl/ocl_engine_factory.hpp"
#include "ze/ze_engine_factory.hpp"
#ifdef OV_GPU_WITH_SYCL_RT
#include "sycl/sycl_engine_factory.hpp"
#endif  // OV_GPU_WITH_SYCL_RT

#include <string>
#include <vector>
#include <memory>
#include <set>
#include <stdexcept>
#include <algorithm>

#if defined(_WIN32)
# ifndef NOMINMAX
#  define NOMINMAX
# endif
# include <windows.h>


#ifdef ENABLE_GTPIN_INTEGRATION
    #include "gtpin_api.h"
    #include <iostream>
    #include <mutex>
    using namespace gtpin;
#endif


static size_t get_cpu_ram_size() {
    MEMORYSTATUSEX s {};
    s.dwLength = sizeof(s);
    GlobalMemoryStatusEx(&s);
    return s.ullTotalPhys;
}
#elif defined(__APPLE__) || defined(__FreeBSD__) || defined(__QNXNTO__)
# include <unistd.h>
# include <sys/sysctl.h>

static size_t get_cpu_ram_size() {
# ifdef __APPLE__
    int query_ram[] = {CTL_HW, HW_MEMSIZE};
# else
    int query_ram[] = {CTL_HW, HW_PHYSMEM};
# endif
    int query_ram_len = sizeof(query_ram) / sizeof(*query_ram);
    size_t totalram = 0;
    size_t length = sizeof(totalram);

    sysctl(query_ram, query_ram_len, &totalram, &length, NULL, 0);
    return totalram;
}
#else
# include <sys/sysinfo.h>

static size_t get_cpu_ram_size() {
    struct sysinfo s {};
    sysinfo(&s);
    return s.totalram;
}
#endif

namespace cldnn {

engine::engine(const device::ptr device)
    : _device(device) {}

const device_info& engine::get_device_info() const {
    return _device->get_info();
}

const device::ptr engine::get_device() const {
    return _device;
}

bool engine::use_unified_shared_memory() const {
    GPU_DEBUG_IF(ExecutionConfig::get_disable_usm()) {
        return false;
    }
    return _device->get_mem_caps().supports_usm();
}

uint64_t engine::get_max_memory_size() const {
    static uint64_t max_device_mem = get_host_memory_size();
    const auto& dev_type = get_device_info().dev_type;
    if (dev_type == device_type::discrete_gpu) {
        max_device_mem += get_device_info().max_global_mem_size;
    }
    return max_device_mem;
}

uint64_t engine::get_host_memory_size() const {
    return static_cast<uint64_t>(get_cpu_ram_size());
}

bool engine::supports_allocation(allocation_type type) const {
    if (memory_capabilities::is_usm_type(type) && !use_unified_shared_memory())
        return false;
    if (allocation_type::usm_shared == type)
        return false;
    return _device->get_mem_caps().support_allocation_type(type);
}

bool engine::can_use_host_usm_zero_copy() const {
    const auto& info = get_device_info();
    return info.dev_type == cldnn::device_type::integrated_gpu && info.arch >= cldnn::gpu_arch::xe2 && supports_allocation(cldnn::allocation_type::usm_host);
}

allocation_type engine::get_lockable_preferred_memory_allocation_type(bool is_image_layout) const {
    if (!use_unified_shared_memory() || is_image_layout)
        return get_default_allocation_type();

    /*
        We do not check device allocation here.
        Device allocation is reserved for buffers of hidden layers.
        Const buffers are propagated to device if possible.
    */

    bool support_usm_host = supports_allocation(allocation_type::usm_host);
    bool support_usm_shared = supports_allocation(allocation_type::usm_shared);

    if (support_usm_shared)
        return allocation_type::usm_shared;
    if (support_usm_host)
        return allocation_type::usm_host;

    OPENVINO_ASSERT(false, "[GPU] Couldn't find proper allocation type in get_lockable_preferred_memory_allocation_type method");
}

allocation_type engine::get_preferred_memory_allocation_type(bool is_image_layout) const {
    if (!use_unified_shared_memory() || is_image_layout)
        return get_default_allocation_type();

    if (supports_allocation(allocation_type::usm_device))
        return allocation_type::usm_device;

    // Fallback to host allocations in case if device ones are not supported for some reason
    if (supports_allocation(allocation_type::usm_host))
        return allocation_type::usm_host;

    OPENVINO_ASSERT(false, "[GPU] Couldn't find proper allocation type in get_preferred_memory_allocation_type method");
}

memory::ptr engine::attach_memory(const layout& layout, void* ptr) {
    return std::make_shared<simple_attached_memory>(layout, ptr);
}

memory::ptr engine::allocate_memory(const layout& layout, bool reset) {
    allocation_type type = get_lockable_preferred_memory_allocation_type(layout.format.is_image_2d());
    return allocate_memory(layout, type, reset);
}

memory_ptr engine::share_buffer(const layout& layout, shared_handle buf) {
    shared_mem_params params = { shared_mem_type::shared_mem_buffer, nullptr, nullptr, buf,
#ifdef _WIN32
        nullptr,
#else
        0,
#endif
        0 };
    return reinterpret_handle(layout, params);
}

memory_ptr engine::share_usm(const layout& layout, shared_handle usm_ptr) {
    shared_mem_params params = { shared_mem_type::shared_mem_usm, nullptr, nullptr, usm_ptr,
#ifdef _WIN32
        nullptr,
#else
        0,
#endif
        0 };
    return reinterpret_handle(layout, params);
}

memory::ptr engine::share_image(const layout& layout, shared_handle img) {
    shared_mem_params params = { shared_mem_type::shared_mem_image, nullptr, nullptr, img,
#ifdef _WIN32
        nullptr,
#else
        0,
#endif
        0 };
    return reinterpret_handle(layout, params);
}

#ifdef _WIN32
memory_ptr engine::share_surface(const layout& layout, shared_handle surf, uint32_t plane) {
    shared_mem_params params = { shared_mem_type::shared_mem_vasurface, nullptr, nullptr, nullptr, surf, plane };
    return reinterpret_handle(layout, params);
}

memory_ptr engine::share_dx_buffer(const layout& layout, shared_handle res) {
    shared_mem_params params = { shared_mem_type::shared_mem_dxbuffer, nullptr, nullptr, res, nullptr, 0 };
    return reinterpret_handle(layout, params);
}
#else
memory_ptr engine::share_surface(const layout& layout, shared_surface surf, uint32_t plane) {
    shared_mem_params params = { shared_mem_type::shared_mem_vasurface, nullptr, nullptr, nullptr, surf, plane };
    return reinterpret_handle(layout, params);
}
#endif  // _WIN32

uint64_t engine::get_max_used_device_memory() const {
    uint64_t total_peak_memory_usage {0};
    for (auto const& m : _peak_memory_usage_data) {
        total_peak_memory_usage += m.load();
    }
    return total_peak_memory_usage;
}

uint64_t engine::get_max_used_device_memory(allocation_type type) const {
    return _peak_memory_usage_data[static_cast<size_t>(type)].load();
}

uint64_t engine::get_used_device_memory(allocation_type type) const {
    return _memory_usage_data[static_cast<size_t>(type)].load();
}

std::map<std::string, uint64_t> engine::get_memory_statistics() const {
    std::map<std::string, uint64_t> statistics;
    const auto add_stat = [&](allocation_type type) {
        auto idx = static_cast<size_t>(type);
        auto value = _memory_usage_data[idx].load();
        std::ostringstream oss;
        oss << type;
        statistics[oss.str()] = value;
    };

    add_stat(allocation_type::unknown);
    add_stat(allocation_type::cl_mem);
    add_stat(allocation_type::usm_host);
    add_stat(allocation_type::usm_shared);
    add_stat(allocation_type::usm_device);
    return statistics;
}

void engine::add_memory_used(uint64_t bytes, allocation_type type) {
    auto idx = static_cast<size_t>(type);
    const auto new_val = _memory_usage_data[idx].fetch_add(bytes) + bytes;
    // Make sure actual maximum value is stored
    while (new_val > _peak_memory_usage_data[idx]) {
        _peak_memory_usage_data[idx] = new_val;
    }
}

void engine::subtract_memory_used(uint64_t bytes, allocation_type type) {
    auto idx = static_cast<size_t>(type);
    if (_memory_usage_data[idx].load() < bytes) {
        throw std::runtime_error("Attempt to free unallocated memory");
    }
    _memory_usage_data[idx] -= bytes;
}

void engine::set_enable_large_allocations(bool enable_large_allocations) {
    this->enable_large_allocations = enable_large_allocations;
}

bool engine::get_enable_large_allocations() const {
    return enable_large_allocations;
}


// Temporary GTPin runtime + external tool DLL experiment.
// Goal:
//   1. Use the already-linked GTPin runtime to obtain IGtCore.
//   2. Load the external funtime sample tool DLL directly.
//   3. Resolve and call GTPin_Entry so the tool registers itself.
//   4. Keep the tool DLL loaded for the rest of the process lifetime.
#ifdef ENABLE_GTPIN_INTEGRATION

namespace {

static std::once_flag g_gtpin_once;
static gtpin::IGtCore* g_gtpin_core = nullptr;
static HMODULE g_funtime_tool = nullptr;

std::string windows_error_message(DWORD error) {
    if (error == 0) {
        return "No error";
    }

    LPSTR buffer = nullptr;
    DWORD size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER |
                                    FORMAT_MESSAGE_FROM_SYSTEM |
                                    FORMAT_MESSAGE_IGNORE_INSERTS,
                                nullptr,
                                error,
                                MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                                reinterpret_cast<LPSTR>(&buffer),
                                0,
                                nullptr);

    std::string message = buffer ? std::string(buffer, size) : "Unknown Windows loader error";
    if (buffer) {
        LocalFree(buffer);
    }
    return message;
}

bool file_exists_w(const wchar_t* path) {
    DWORD attrs = GetFileAttributesW(path);
    return attrs != INVALID_FILE_ATTRIBUTES && !(attrs & FILE_ATTRIBUTE_DIRECTORY);
}

bool directory_exists_w(const wchar_t* path) {
    DWORD attrs = GetFileAttributesW(path);
    return attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY);
}

void print_current_directory() {
    wchar_t buffer[MAX_PATH] = {};
    DWORD size = GetCurrentDirectoryW(MAX_PATH, buffer);
    if (size == 0 || size >= MAX_PATH) {
        std::cout << "[GTPIN] Failed to read current directory. Error "
                  << GetLastError() << ": " << windows_error_message(GetLastError()) << std::endl;
        return;
    }

    std::wcout << L"[GTPIN] Current directory: " << buffer << std::endl;
}

void print_module_path(const wchar_t* module_name) {
    HMODULE module = GetModuleHandleW(module_name);

    std::wcout << L"[GTPIN] GetModuleHandleW(" << module_name << L") = "
               << module << std::endl;

    if (!module) {
        return;
    }

    wchar_t module_path[MAX_PATH] = {};
    DWORD size = GetModuleFileNameW(module, module_path, MAX_PATH);
    if (size == 0 || size >= MAX_PATH) {
        std::cout << "[GTPIN] Failed to read module path for module. Error "
                  << GetLastError() << ": " << windows_error_message(GetLastError()) << std::endl;
        return;
    }

    std::wcout << L"[GTPIN] Module path: " << module_path << std::endl;
}

void print_path_hint() {
    DWORD required = GetEnvironmentVariableW(L"PATH", nullptr, 0);
    std::wcout << L"[GTPIN] PATH length: " << required << std::endl;

    if (required == 0 || required > 32767) {
        return;
    }

    std::wstring path(required, L'\0');
    GetEnvironmentVariableW(L"PATH", path.data(), required);

    // Avoid dumping the entire PATH into the sample output.
    std::wcout << L"[GTPIN] PATH prefix: " << path.substr(0, 800) << std::endl;
}

bool add_dll_search_directory(const wchar_t* dir) {
    std::wcout << L"[GTPIN] Adding DLL search directory via SetDllDirectoryW: "
               << dir << std::endl;

    if (!SetDllDirectoryW(dir)) {
        DWORD error = GetLastError();
        std::cout << "[GTPIN] SetDllDirectoryW failed. Error "
                  << error << ": " << windows_error_message(error) << std::endl;
        return false;
    }

    return true;
}

bool initialize_gtpin_runtime() {
    if (g_gtpin_core) {
        std::cout << "[GTPIN] Runtime already initialized. Core="
                  << g_gtpin_core << std::endl;
        return true;
    }

    std::cout << "[GTPIN] Initializing runtime through already-linked GTPin_GetCore()" << std::endl;

    g_gtpin_core = GTPin_GetCore();

    std::cout << "[GTPIN] GTPin_GetCore returned core="
              << g_gtpin_core << std::endl;

    if (!g_gtpin_core) {
        std::cout << "[GTPIN] Runtime initialization failed: core is null" << std::endl;
        return false;
    }

    print_module_path(L"gtpin.dll");
    print_module_path(L"gtpin_core.dll");
    print_module_path(L"iga_wrapper.dll");
    print_module_path(L"ged.dll");

    return true;
}

bool load_gtpin_tool_dll() {
    if (!g_gtpin_core) {
        std::cout << "[GTPIN] Cannot load tool: runtime core is null" << std::endl;
        return false;
    }

    if (g_funtime_tool) {
        std::cout << "[GTPIN] funtime tool already loaded. HMODULE="
                  << g_funtime_tool << std::endl;
        return true;
    }

    const wchar_t* gtpin_lib_dir =
        L"W:\\Building\\GSOC\\external-release-gtpin-4.7.1-win\\Profilers\\Lib\\intel64";

    const wchar_t* funtime_dir =
        L"W:\\Building\\GSOC\\external-release-gtpin-4.7.1-win\\Profilers\\Examples\\intel64";

    const wchar_t* funtime_path =
        L"W:\\Building\\GSOC\\external-release-gtpin-4.7.1-win\\Profilers\\Examples\\intel64\\funtime.dll";

    std::cout << "[GTPIN] === Tool DLL loading diagnostics ===" << std::endl;

    print_current_directory();
    print_path_hint();

    std::wcout << L"[GTPIN] GTPin lib dir: " << gtpin_lib_dir
               << L" exists=" << (directory_exists_w(gtpin_lib_dir) ? L"YES" : L"NO")
               << std::endl;

    std::wcout << L"[GTPIN] funtime dir: " << funtime_dir
               << L" exists=" << (directory_exists_w(funtime_dir) ? L"YES" : L"NO")
               << std::endl;

    std::wcout << L"[GTPIN] funtime DLL: " << funtime_path
               << L" exists=" << (file_exists_w(funtime_path) ? L"YES" : L"NO")
               << std::endl;

    print_module_path(L"gtpin.dll");

    // funtime.dll depends on gtpin.dll + KERNEL32.dll.
    // Make both the runtime directory and the tool directory discoverable for the Windows loader.
    add_dll_search_directory(gtpin_lib_dir);
    add_dll_search_directory(funtime_dir);

    SetLastError(0);

    std::wcout << L"[GTPIN] Loading funtime with LoadLibraryExW: "
               << funtime_path << std::endl;

    g_funtime_tool = LoadLibraryExW(funtime_path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);

    if (!g_funtime_tool) {
        DWORD error = GetLastError();

        std::cout << "[GTPIN] LoadLibraryExW(funtime.dll) failed." << std::endl;
        std::cout << "[GTPIN] Error code: " << error << std::endl;
        std::cout << "[GTPIN] Error message: " << windows_error_message(error) << std::endl;

        if (error == ERROR_MOD_NOT_FOUND) {
            std::cout << "[GTPIN] Interpretation: target DLL path is wrong, or a dependent DLL "
                         "such as gtpin.dll was not found by the Windows loader."
                      << std::endl;
        } else if (error == ERROR_BAD_EXE_FORMAT) {
            std::cout << "[GTPIN] Interpretation: architecture mismatch, e.g. x86 DLL in x64 process."
                      << std::endl;
        } else if (error == ERROR_PROC_NOT_FOUND) {
            std::cout << "[GTPIN] Interpretation: dependency was found, but a required export was missing."
                      << std::endl;
        }

        return false;
    }

    std::cout << "[GTPIN] funtime.dll loaded successfully. HMODULE="
              << g_funtime_tool << std::endl;

    using GTPinEntryFn = void (*)(int, const char**);

    SetLastError(0);

    auto entry = reinterpret_cast<GTPinEntryFn>(
        GetProcAddress(g_funtime_tool, "GTPin_Entry"));

    if (!entry) {
        DWORD error = GetLastError();

        std::cout << "[GTPIN] Failed to resolve GTPin_Entry." << std::endl;
        std::cout << "[GTPIN] Error code: " << error << std::endl;
        std::cout << "[GTPIN] Error message: " << windows_error_message(error) << std::endl;

        return false;
    }

    std::cout << "[GTPIN] Resolved GTPin_Entry="
              << reinterpret_cast<void*>(entry) << std::endl;


    std::cout << "[GTPIN] Calling funtime GTPin_Entry(argc=0, argv=nullptr)" << std::endl;

    //failing here!!
    entry(0, nullptr);

    std::cout << "[GTPIN] GTPin_Entry returned successfully" << std::endl;

    return true;
}

void initialize_gtpin_once() {
    std::call_once(g_gtpin_once, [] {
        std::cout << "[GTPIN] === Embedded runtime/tool experiment start ===" << std::endl;

        if (!initialize_gtpin_runtime()) {
            std::cout << "[GTPIN] Runtime initialization failed" << std::endl;
            return;
        }

        if (!load_gtpin_tool_dll()) {
            std::cout << "[GTPIN] Tool loading failed" << std::endl;
            return;
        }

        std::cout << "[GTPIN] Runtime + external funtime tool initialization completed"
                  << std::endl;
    });
}

}  // namespace

#endif





//J--Is this the GPU runtime creation spot?
std::shared_ptr<cldnn::engine> engine::create(engine_types engine_type, runtime_types runtime_type, const device::ptr device) {
    std::shared_ptr<cldnn::engine> ret;

//GTPin tool registration
std::cout << "[Runtime-GTPin] before GTPin registration" << std::endl;
#ifdef ENABLE_GTPIN_INTEGRATION
    initialize_gtpin_once();
#endif
    std::cout << "[Runtime-GTPin] after GTPin registration" << std::endl;
    std::cout << "[Runtime-GTPin] before create_ocl_engine" << std::endl;


    switch (engine_type) {
#ifdef OV_GPU_WITH_SYCL_RT
    case engine_types::sycl:
        ret = sycl::create_sycl_engine(device, runtime_type);
        break;
#endif  // OV_GPU_WITH_SYCL_RT
#ifdef OV_GPU_WITH_OCL_RT
#ifdef OV_GPU_WITH_SYCL
    case engine_types::sycl:
        ret = ocl::create_sycl_engine(device, runtime_type);
        break;
#endif  // OV_GPU_WITH_SYCL
    case engine_types::ocl:
        ret = ocl::create_ocl_engine(device, runtime_type);
        break;
#endif
#ifdef OV_GPU_WITH_ZE_RT
    case engine_types::ze:
        ret = ze::create_ze_engine(device, runtime_type);
        break;
#endif
    default:
        throw std::runtime_error("Invalid engine type");
    }


    std::cout << "[Runtime-GTPin] after create_ocl_engine" << std::endl;


    const auto& info = device->get_info();
    GPU_DEBUG_INFO << "Selected Device: " << info.dev_name << std::endl;
    return ret;
}


//The second candidate before the device_query


std::shared_ptr<cldnn::engine> engine::create(engine_types engine_type, runtime_types runtime_type) {
    device_query query(engine_type, runtime_type, nullptr, nullptr, 0, -1, true);
    auto devices = query.get_available_devices();

    OPENVINO_ASSERT(!devices.empty(), "[GPU] Can't create ", engine_type, " engine for ", runtime_type, " runtime as no suitable devices are found\n"
                                      "[GPU] Please check OpenVINO documentation for GPU drivers setup guide.\n");

    auto iter = devices.find(std::to_string(device_query::device_id));
    auto& device = iter != devices.end() ? iter->second : devices.begin()->second;

    return engine::create(engine_type, runtime_type, device);
}

bool engine::check_allocatable(const layout& layout, allocation_type type) {
    OPENVINO_ASSERT(supports_allocation(type), "[GPU] Unsupported allocation type: ", type);

    if (!get_enable_large_allocations()) {
        bool exceed_allocatable_mem_size = (layout.bytes_count() > get_device_info().max_alloc_mem_size);

        // When dynamic shape upper bound makes bigger buffer, then return false.
        if (exceed_allocatable_mem_size && layout.is_dynamic()) {
            OPENVINO_ASSERT(layout.has_upper_bound(), "[GPU] Dynamic shape without upper bound tries to allocate");
            return false;
        }

        OPENVINO_ASSERT(!exceed_allocatable_mem_size,
                        "[GPU] Exceeded max size of memory object allocation: ",
                        "requested ", layout.bytes_count(), " bytes, "
                        "but max alloc size supported by device is ", get_device_info().max_alloc_mem_size, " bytes. ",
                        "Please try to reduce batch size, use lower precision, "
                        "or set ov::intel_gpu::hint::enable_large_allocations config property to true.");
    }

    auto used_mem = get_used_device_memory(allocation_type::usm_device) + get_used_device_memory(allocation_type::usm_host);
    auto exceed_available_mem_size = (layout.bytes_count() + used_mem > get_max_memory_size());

    // When dynamic shape upper bound makes bigger buffer, then return false.
    if (exceed_available_mem_size && layout.is_dynamic()) {
        OPENVINO_ASSERT(layout.has_upper_bound(), "[GPU] Dynamic shape without upper bound tries to allocate");
        return false;
    }

#ifdef __unix__
    // Prevent from being killed by Ooo Killer of Linux
    OPENVINO_ASSERT(!exceed_available_mem_size,
                    "[GPU] Exceeded max size of memory allocation: ",
                    "Required ", layout.bytes_count(), " bytes, already occupied : ", used_mem, " bytes, ",
                    "but available memory size is ", get_max_memory_size(), " bytes");
#else
    if (exceed_available_mem_size) {
        GPU_DEBUG_COUT << "[Warning] [GPU] Exceeded max size of memory allocation: " << "Required " << layout.bytes_count() << " bytes, already occupied : "
                       << used_mem << " bytes, but available memory size is " << get_max_memory_size() << " bytes" << std::endl;
        GPU_DEBUG_COUT << "Please note that performance might drop due to memory swap." << std::endl;
    }
#endif

    return true;
}

#ifdef ENABLE_ONEDNN_FOR_GPU
dnnl::engine& engine::get_onednn_engine() const {
    const std::lock_guard<std::mutex> lock(onednn_mutex);
    OPENVINO_ASSERT(_onednn_engine, "[GPU] Can't get onednn engine handle as it was not initialized. Please check that create_onednn_engine() was called");
    return *_onednn_engine;
}
#endif

stream& engine::get_service_stream() const {
    return *_service_stream;
}

}  // namespace cldnn
