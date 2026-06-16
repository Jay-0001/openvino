#pragma once

#include <memory>
#include <string>
#include <vector>

namespace ov::intel_gpu::gtpin {

class SharedLibrary;

struct GtpinOptions {
    std::string main_library_path;
    std::string tool_path;
    std::vector<std::string> dependency_paths;
    bool external_mode = false;
};

class GtpinSession {
public:
    ~GtpinSession();

    static std::unique_ptr<GtpinSession>
    create(const GtpinOptions& options);

    bool valid() const;

private:
    GtpinSession();

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}  // namespace ov::intel_gpu::gtpin