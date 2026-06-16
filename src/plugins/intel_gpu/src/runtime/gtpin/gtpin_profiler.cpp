// runtime/gtpin/gtpin_profiler.cpp

#include "gtpin_profiler.hpp"
#include "gtpin_session.hpp"

#include <iostream>
#include <memory>
#include <mutex>
#include <string>

namespace ov::intel_gpu::gtpin {

namespace {

std::once_flag g_gtpin_once;
std::unique_ptr<GtpinProfiler> g_gtpin_profiler;

}  // namespace

GtpinProfiler::GtpinProfiler() = default;

bool GtpinProfiler::initialize() {
    const char* runtime_dir = std::getenv("OV_GTPIN_RUNTIME_DIR");
    const char* tool_path = std::getenv("OV_GTPIN_TOOL_PATH");

    if (!runtime_dir || !tool_path) {
        std::cerr
            << "[OV][GTPIN] Required environment variables not set"
            << std::endl;
        return false;
    }

    std::string gtpin_lib_dir(runtime_dir);
    std::string gtpin_tool_path(tool_path);
    
    GtpinOptions options;

    options.dependency_paths = {
        gtpin_lib_dir + R"(\ged.dll)",
        gtpin_lib_dir + R"(\gtpin_core.dll)",
        gtpin_lib_dir + R"(\iga_wrapper.dll)"
    };

    options.main_library_path = gtpin_lib_dir + R"(\gtpin.dll)";
    options.tool_path = gtpin_tool_path;
    options.external_mode = true;

    std::cerr << "[OV][GTPIN] Initializing GTPin profiler" << std::endl;

    m_session = GtpinSession::create(options);

    if (!m_session || !m_session->valid()) {
        std::cerr << "[OV][GTPIN] Failed to initialize GTPin session" << std::endl;
        m_session.reset();
        return false;
    }

    std::cerr << "[OV][GTPIN] GTPin profiler initialized successfully" << std::endl;
    return true;
}

bool GtpinProfiler::enabled() const {
    return m_session && m_session->valid();
}

void initialize_once() {
    std::call_once(g_gtpin_once, [] {
        g_gtpin_profiler = std::make_unique<GtpinProfiler>();

        if (!g_gtpin_profiler->initialize()) {
            std::cerr << "[OV][GTPIN] initialize_once failed" << std::endl;
            g_gtpin_profiler.reset();
        }
    });

    if (g_gtpin_profiler)
        std::cout<< "[OV][GTPIN] GTPin session initialized"<< std::endl;
}

}  // namespace ov::intel_gpu::gtpin