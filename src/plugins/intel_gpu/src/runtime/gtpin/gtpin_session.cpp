#include "gtpin_session.hpp"
#include "gtpin_shared_library.hpp"

#include <filesystem>
#include <iostream>

#include "gtpin_api.h"

namespace ov::intel_gpu::gtpin {

namespace {

using GTPinGetCoreFn = ::gtpin::IGtCore* (*)();
using GTPinEntryFn = void (*)(int, const char**);

}  // namespace

struct GtpinSession::Impl {
    std::vector<std::unique_ptr<SharedLibrary>> dependencies;

    std::unique_ptr<SharedLibrary> gtpin_library;
    std::unique_ptr<SharedLibrary> tool_library;

    ::gtpin::IGtCore* core = nullptr;
    ::gtpin::GtToolHandle tool_handle = nullptr;

    bool keep_core_for_external_tool = false;
};

GtpinSession::GtpinSession()
    : m_impl(new Impl()) {}

GtpinSession::~GtpinSession() = default;

std::unique_ptr<GtpinSession>
GtpinSession::create(const GtpinOptions& options) {
    auto session = std::unique_ptr<GtpinSession>(new GtpinSession());
    auto& impl = *session->m_impl;

    if (options.main_library_path.empty()) {
        std::cerr << "[OV][GTPIN] main_library_path is empty" << std::endl;
        return nullptr;
    }

    if (!std::filesystem::exists(options.main_library_path)) {
        std::cerr << "[OV][GTPIN] GTPin runtime library does not exist: "
                  << options.main_library_path
                  << std::endl;
        return nullptr;
    }

    for (const auto& dependency_path : options.dependency_paths) {
        if (dependency_path.empty()) {
            continue;
        }

        if (!std::filesystem::exists(dependency_path)) {
            std::cerr << "[OV][GTPIN] Dependency library does not exist: "
                      << dependency_path
                      << std::endl;
            return nullptr;
        }

        auto dependency_result = SharedLibrary::load(
            dependency_path,
            false,
            !options.external_mode);

        if (!dependency_result.second || !dependency_result.first) {
            std::cerr << "[OV][GTPIN] Failed to load dependency: "
                      << dependency_path
                      << std::endl;
            return nullptr;
        }

        impl.dependencies.push_back(std::move(dependency_result.first));
    }

    auto gtpin_result = SharedLibrary::load(
        options.main_library_path,
        options.external_mode,
        !options.external_mode);

    if (!gtpin_result.second || !gtpin_result.first) {
        std::cerr << "[OV][GTPIN] Failed to load GTPin runtime library"
                  << std::endl;
        return nullptr;
    }

    void* get_core_symbol = gtpin_result.first->symbol("GTPin_GetCore");
    if (!get_core_symbol) {
        std::cerr << "[OV][GTPIN] Could not resolve GTPin_GetCore"
                  << std::endl;
        return nullptr;
    }

    auto get_core = reinterpret_cast<GTPinGetCoreFn>(get_core_symbol);
    impl.core = get_core();

    if (!impl.core) {
        std::cerr << "[OV][GTPIN] GTPin_GetCore returned nullptr"
                  << std::endl;
        return nullptr;
    }

    std::cerr << "[OV][GTPIN] Loaded GTPin core. Version="
              << impl.core->Version()
              << ", Revision="
              << impl.core->Revision()
              << ", API="
              << impl.core->ApiVersion()
              << std::endl;

    impl.gtpin_library = std::move(gtpin_result.first);

    if (!options.tool_path.empty()) {
        if (!std::filesystem::exists(options.tool_path)) {
            std::cerr << "[OV][GTPIN] Tool library does not exist: "
                      << options.tool_path
                      << std::endl;
            return nullptr;
        }

        auto tool_result = SharedLibrary::load(
            options.tool_path,
            false,
            false);

        if (!tool_result.second || !tool_result.first) {
            std::cerr << "[OV][GTPIN] Failed to load GTPin tool library: "
                      << options.tool_path
                      << std::endl;
            return nullptr;
        }

        void* entry_symbol = tool_result.first->symbol("GTPin_Entry");
        if (!entry_symbol) {
            std::cerr << "[OV][GTPIN] Could not resolve GTPin_Entry from tool: "
                      << options.tool_path
                      << std::endl;
            return nullptr;
        }

        auto entry = reinterpret_cast<GTPinEntryFn>(entry_symbol);

        std::cerr << "[OV][GTPIN] Calling GTPin_Entry for external tool"
                  << std::endl;

        entry(0, nullptr);

        impl.tool_library = std::move(tool_result.first);
        impl.keep_core_for_external_tool = true;

        std::cerr << "[OV][GTPIN] External GTPin tool initialized"
                  << std::endl;
    }

    return session;
}

bool GtpinSession::valid() const {
    return m_impl && m_impl->core;
}


}  // namespace ov::intel_gpu::gtpin