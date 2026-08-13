// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/variable_state.hpp"
#include "intel_gpu/plugin/output_memory_block.hpp"
#include "intel_gpu/primitives/read_value.hpp"
#include "intel_gpu/primitives/lora.hpp"
#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"
#include "intel_gpu/primitives/input_layout.hpp"

#include "intel_gpu/runtime/error_handler.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/event.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "intel_gpu/runtime/compilation_context.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "openvino/util/env_util.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/graph/serialization/map_serializer.hpp"

#include "primitive_inst.h"
#include "input_layout_inst.h"
#include "fully_connected_inst.h"
#include "paged_attention_inst.h"
#include "convolution_inst.h"
#include "deconvolution_inst.h"
#include "mutable_data_inst.h"
#include "condition_inst.h"
#include "read_value_inst.h"
#include "reshape_inst.h"
#include "kv_cache_inst.h"
#include "program_helpers.h"
#include "program_dump_graph.h"
#include "to_string_utils.h"

#include <algorithm>
#include <string>
#include <vector>
#include <stack>
#include <memory>
#include <set>
#include <utility>
#include <map>
#include <functional>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <atomic>
#include <mutex>

#include "debug_helper.hpp"
#ifdef GPU_DEBUG_CONFIG
#include <fstream>
#include <sys/stat.h>
#include <chrono>
#include <thread>
#include <filesystem>
#endif

namespace cldnn {
namespace {

#ifdef GPU_DEBUG_CONFIG
void dump_perf_data_raw(std::string dump_path, bool per_iter_mode, const std::list<std::shared_ptr<primitive_inst>>& exec_order) {
    auto layouts_to_str = [](const std::vector<layout>& layouts) -> std::string {
        std::stringstream s;
        for (size_t i = 0; i < layouts.size(); i++) {
            s << layouts[i].to_short_string();
            if (i != layouts.size() - 1)
                s << ";";
        }
        return s.str();
    };

    const std::string perf_raw_csv_header = per_iter_mode ? "prim_id,prim_type,stage,net_in_shapes,in_shapes,out_shapes,impl,iter,time_usec\n"
                                                          : "prim_id,prim_type,stage,net_in_shapes,in_shapes,out_shapes,impl,iters,time_usec\n";
    std::ofstream of(dump_path);
    if (of.is_open()) {
        of << perf_raw_csv_header;
        for (auto& inst : exec_order) {
            auto prim_id = inst->id();
            auto& perf_data = inst->get_profiling_data();
            auto& perf_info = inst->get_profiling_info();
            std::vector<size_t> sorted_entries;
            std::transform(perf_data.begin(), perf_data.end(), std::back_inserter(sorted_entries),
            [](const std::pair<size_t, std::tuple<int64_t, size_t>>& e) {
                return e.first;
            });
            std::sort(sorted_entries.begin(), sorted_entries.end(), [&](size_t a, size_t b) -> bool {
                auto& a_info = perf_info.at(a);
                auto& b_info = perf_info.at(b);

                if (a_info.stage != b_info.stage) {
                    return static_cast<std::underlying_type<instrumentation::pipeline_stage>::type>(a_info.stage) <
                           static_cast<std::underlying_type<instrumentation::pipeline_stage>::type>(b_info.stage);
                }

                if (a_info.cache_hit != b_info.cache_hit)
                    return a_info.cache_hit;

                if (a_info.memalloc_info != b_info.memalloc_info)
                    return a_info.memalloc_info.length() < b_info.memalloc_info.length();

                size_t total_out_size_a = 0;
                size_t total_out_size_b = 0;
                for (auto& ol : a_info.output_layouts) {
                    total_out_size_a += ol.count();
                }
                for (auto& ol : b_info.output_layouts) {
                    total_out_size_b += ol.count();
                }
                return total_out_size_a < total_out_size_b;
            });
            for (auto& hash : sorted_entries) {
                auto& key = perf_info.at(hash);
                auto& entry = perf_data.at(hash);
                auto& time = std::get<0>(entry);
                auto num_iters = per_iter_mode ? key.iteration_num : std::get<1>(entry);
                int64_t time_avg = per_iter_mode ? time : time / num_iters;
                std::string net_in_l_str = layouts_to_str(key.network_input_layouts);
                std::string in_l_str = layouts_to_str(key.input_layouts);
                std::string out_l_str = layouts_to_str(key.output_layouts);
                std::string stage_suffix;
                if (key.cache_hit)
                    stage_suffix += " (cache_hit) ";
                if (!key.memalloc_info.empty())
                    stage_suffix += " (" + key.memalloc_info + ") ";
                of << prim_id << ","
                << inst->desc()->type_string() << ","
                << key.stage << stage_suffix << ","
                << net_in_l_str << ","
                << in_l_str << ","
                << out_l_str << ","
                << (key.stage == instrumentation::pipeline_stage::inference ? key.impl_name : "undef") << ","
                << num_iters << ","
                << time_avg << "\n";
            }
        }
    }
}

// Dumps a per-primitive averaged execution time CSV with the same schema as
// benchmark_app --report_type average_counters and the CPU plugin's
// OV_CPU_AVERAGE_COUNTERS feature, so that aggregate-average-counters.py and
// other tooling can be reused across plugins.
void dump_average_counters(std::string dump_path,
                           uint32_t net_id,
                           const std::list<std::shared_ptr<primitive_inst>>& exec_order) {
    if (dump_path.empty())
        return;

    std::filesystem::path file_name{dump_path};
    file_name += "_" + std::to_string(net_id) + ".csv";
    std::ofstream file(file_name);
    if (!file.is_open())
        return;

    const std::string header = "layerName;execStatus;layerType;execType;realTime (ms);cpuTime (ms);";
    file << header << "\n";

    auto to_ms = [](uint64_t value_us) {
        return static_cast<double>(std::chrono::microseconds(value_us).count()) / 1000.0;
    };

    uint64_t total_us = 0;

    for (const auto& inst : exec_order) {
        if (inst->is_constant())
            continue;

        // Aggregate inference-stage entries only, mirroring the CPU plugin which
        // reports just the executed-kernel time. Other GPU pipeline stages
        // (shape_inference, set_arguments, memory_allocation, ...) are host-side
        // overhead that has no CPU-plugin counterpart.
        const auto& perf_data = inst->get_profiling_data();
        const auto& perf_info = inst->get_profiling_info();
        uint64_t prim_total_us = 0;
        size_t prim_total_iters = 0;
        std::string impl_name;
        size_t max_iters_for_impl = 0;
        for (const auto& kv : perf_data) {
            const auto& key = perf_info.at(kv.first);
            if (key.stage != instrumentation::pipeline_stage::inference)
                continue;
            const auto cur_time = static_cast<uint64_t>(std::get<0>(kv.second));
            const auto cur_iters = std::get<1>(kv.second);
            prim_total_us += cur_time;
            prim_total_iters += cur_iters;
            // For dynamic shapes a primitive may have multiple inference entries
            // (one per shape). Pick the impl_name that ran the most iterations
            // as the representative execType.
            if (cur_iters > max_iters_for_impl) {
                max_iters_for_impl = cur_iters;
                impl_name = key.impl_name;
            }
        }

        const uint64_t avg_us = prim_total_iters > 0 ? prim_total_us / prim_total_iters : 0;
        const std::string status = avg_us > 0 ? "EXECUTED" : "NOT_RUN";
        const auto cpu_time = to_ms(avg_us);
        const auto real_time = cpu_time;

        file << inst->id() << ";" << status << ";" << inst->desc()->type_string() << ";"
             << impl_name << ";" << real_time << ";" << cpu_time << ";" << "\n";

        total_us += avg_us;
    }

    const auto total_ms = to_ms(total_us);
    file << "Total;;;;" << total_ms << ";" << total_ms << ";\n";
}

// gtpin integration -- correlation
std::string csv_escape(const std::string& value) {
    if (value.find_first_of("\",\n\r") == std::string::npos) {
        return value;
    }

    std::string escaped = "\"";
    for (char ch : value) {
        if (ch == '"') {
            escaped += "\"\"";
        } else {
            escaped += ch;
        }
    }
    escaped += "\"";
    return escaped;
}

// gtpin integration -- correlation
std::vector<std::string> split_kernel_entries(const std::string& kernel_entries) {
    std::istringstream stream(kernel_entries);
    std::vector<std::string> entries;
    std::string entry;
    while (stream >> entry) {
        entries.push_back(entry);
    }
    return entries;
}

// gsoc gtpin start
std::string argument_type_to_string(argument_desc::Types type) {
    switch (type) {
        case argument_desc::Types::INPUT: return "INPUT";
        case argument_desc::Types::OUTPUT: return "OUTPUT";
        case argument_desc::Types::WEIGHTS: return "WEIGHTS";
        case argument_desc::Types::BIAS: return "BIAS";
        case argument_desc::Types::SCALE_TABLE: return "SCALE_TABLE";
        case argument_desc::Types::SLOPE: return "SLOPE";
        case argument_desc::Types::INTERNAL_BUFFER: return "INTERNAL_BUFFER";
        case argument_desc::Types::SCALAR: return "SCALAR";
        case argument_desc::Types::CELL: return "CELL";
        case argument_desc::Types::WEIGHTS_ZERO_POINTS: return "WEIGHTS_ZERO_POINTS";
        case argument_desc::Types::ACTIVATIONS_ZERO_POINTS: return "ACTIVATIONS_ZERO_POINTS";
        case argument_desc::Types::COMPENSATION: return "COMPENSATION";
        case argument_desc::Types::INPUT_OF_FUSED_PRIMITIVE: return "INPUT_OF_FUSED_PRIMITIVE";
        case argument_desc::Types::SHAPE_INFO: return "SHAPE_INFO";
        case argument_desc::Types::LOCAL_MEMORY_SIZE: return "LOCAL_MEMORY_SIZE";
        default: return "UNKNOWN";
    }
}

bool is_input_memory_argument(argument_desc::Types type) {
    switch (type) {
        case argument_desc::Types::INPUT:
        case argument_desc::Types::WEIGHTS:
        case argument_desc::Types::BIAS:
        case argument_desc::Types::SCALE_TABLE:
        case argument_desc::Types::SLOPE:
        case argument_desc::Types::INTERNAL_BUFFER:
        case argument_desc::Types::CELL:
        case argument_desc::Types::WEIGHTS_ZERO_POINTS:
        case argument_desc::Types::ACTIVATIONS_ZERO_POINTS:
        case argument_desc::Types::COMPENSATION:
        case argument_desc::Types::INPUT_OF_FUSED_PRIMITIVE:
        case argument_desc::Types::SHAPE_INFO:
            return true;
        default:
            return false;
    }
}

bool is_output_memory_argument(argument_desc::Types type) {
    return type == argument_desc::Types::OUTPUT;
}

memory::cptr resolve_argument_memory(const argument_desc& arg_desc, const kernel_arguments_data& args) {
    switch (arg_desc.t) {
        case argument_desc::Types::INPUT:
            return arg_desc.index < args.inputs.size() ? args.inputs[arg_desc.index] : nullptr;
        case argument_desc::Types::OUTPUT:
            return arg_desc.index < args.outputs.size() ? args.outputs[arg_desc.index] : nullptr;
        case argument_desc::Types::WEIGHTS:
            return args.weights;
        case argument_desc::Types::BIAS:
            return args.bias;
        case argument_desc::Types::SCALE_TABLE:
            return args.scale_table;
        case argument_desc::Types::SLOPE:
            return args.slope;
        case argument_desc::Types::INTERNAL_BUFFER:
            return arg_desc.index < args.intermediates.size() ? args.intermediates[arg_desc.index] : nullptr;
        case argument_desc::Types::CELL:
            return args.cell;
        case argument_desc::Types::WEIGHTS_ZERO_POINTS:
            return args.weights_zero_points;
        case argument_desc::Types::ACTIVATIONS_ZERO_POINTS:
            return args.activations_zero_points;
        case argument_desc::Types::COMPENSATION:
            return args.compensation;
        case argument_desc::Types::INPUT_OF_FUSED_PRIMITIVE:
            return arg_desc.index < args.fused_op_inputs.size() ? args.fused_op_inputs[arg_desc.index] : nullptr;
        case argument_desc::Types::SHAPE_INFO:
            return args.shape_info;
        default:
            return nullptr;
    }
}

std::string get_memory_identity_string(const memory::cptr& mem) {
    if (!mem) {
        return "";
    }

    const auto params = mem->get_internal_params();
    const void* identity = params.mem != nullptr ? params.mem : mem->buffer_ptr();
    if (!identity) {
        return "";
    }

    std::ostringstream output;
    output << identity;
    return output.str();
}

std::string get_kernel_input_arg_addresses(const kernel_arguments_desc& args_desc, const kernel_arguments_data& args) {
    std::vector<std::string> formatted_args;

    for (size_t ordinal = 0; ordinal < args_desc.arguments.size(); ++ordinal) {
        const auto& arg_desc = args_desc.arguments[ordinal];
        if (!is_input_memory_argument(arg_desc.t)) {
            continue;
        }

        const auto mem = resolve_argument_memory(arg_desc, args);
        std::ostringstream entry;
        entry << ordinal << ":" << argument_type_to_string(arg_desc.t) << ":" << arg_desc.index << ":"
              << (mem ? mem->get_allocation_type() : allocation_type::unknown) << ":"
              << get_memory_identity_string(mem);
        formatted_args.push_back(entry.str());
    }

    std::ostringstream output;
    for (size_t i = 0; i < formatted_args.size(); ++i) {
        if (i != 0) {
            output << ";";
        }
        output << formatted_args[i];
    }
    return output.str();
}

std::string get_kernel_output_arg_addresses(const kernel_arguments_desc& args_desc, const kernel_arguments_data& args) {
    std::vector<std::string> formatted_args;

    for (size_t ordinal = 0; ordinal < args_desc.arguments.size(); ++ordinal) {
        const auto& arg_desc = args_desc.arguments[ordinal];
        if (!is_output_memory_argument(arg_desc.t)) {
            continue;
        }

        const auto mem = resolve_argument_memory(arg_desc, args);
        std::ostringstream entry;
        entry << ordinal << ":" << argument_type_to_string(arg_desc.t) << ":" << arg_desc.index << ":"
              << (mem ? mem->get_allocation_type() : allocation_type::unknown) << ":"
              << get_memory_identity_string(mem);
        formatted_args.push_back(entry.str());
    }

    std::ostringstream output;
    for (size_t i = 0; i < formatted_args.size(); ++i) {
        if (i != 0) {
            output << ";";
        }
        output << formatted_args[i];
    }
    return output.str();
}

std::string get_output_memory_addresses(const kernel_arguments_data& args) {
    std::vector<std::string> formatted_args;
    std::set<std::string> seen_entries;

    for (size_t output_index = 0; output_index < args.outputs.size(); ++output_index) {
        const auto& mem = args.outputs[output_index];
        std::ostringstream entry;
        entry << output_index << ":OUTPUT_BUFFER:"
              << (mem ? mem->get_allocation_type() : allocation_type::unknown) << ":"
              << get_memory_identity_string(mem);
        if (seen_entries.insert(entry.str()).second) {
            formatted_args.push_back(entry.str());
        }
    }

    std::ostringstream output;
    for (size_t i = 0; i < formatted_args.size(); ++i) {
        if (i != 0) {
            output << ";";
        }
        output << formatted_args[i];
    }
    return output.str();
}

std::mutex& get_dispatch_dump_mutex() {
    static std::mutex dispatch_dump_mutex;
    return dispatch_dump_mutex;
}

uint64_t acquire_global_dispatch_dump_id() {
    static std::atomic<uint64_t> global_dispatch_id{0};
    return global_dispatch_id.fetch_add(1, std::memory_order_relaxed);
}
// gsoc gtpin end

// gsoc gtpin
template <typename Container>
std::string join_strings(const Container& values, const std::string& separator) {
    std::ostringstream output;
    bool first = true;
    for (const auto& value : values) {
        if (!first) {
            output << separator;
        }
        output << value;
        first = false;
    }
    return output.str();
}

struct topdown_primitive_row {
    std::string origin_op_name;
    std::string primitive_id;
    std::string original_primitive_id;
    std::string primitive_type;
    std::string implementation;
    int exec_id = -1;
    bool is_input = false;
    bool is_output = false;
    std::vector<std::string> dependencies;
    std::vector<std::string> users;
    std::vector<std::string> fused_ids;
};

struct topdown_summary_row {
    std::set<std::string> primitive_ids;
    std::set<std::string> original_primitive_ids;
    std::set<std::string> primitive_types;
    std::set<std::string> implementations;
    std::set<std::string> optimized_out_ids;
};
// gsoc gtpin

#else
void dump_perf_data_raw(std::string, bool per_iter_mode, const std::list<std::shared_ptr<primitive_inst>>&) {}
void dump_average_counters(std::string, uint32_t, const std::list<std::shared_ptr<primitive_inst>>&) {}
#endif
}  // namespace

static uint32_t get_unique_net_id() {
    static std::atomic<uint32_t> id_gen{0};
    return ++id_gen;
}

/*
Network will always have net_id = 0 when it will be cldnn internal micronetwork (created i.e by propagate_constants
opt pass).
*/
network::network(program::ptr program, stream::ptr stream, bool is_internal, bool is_primary_stream)
    : _program(program)
    , _engine(program->get_engine())
    , _stream(stream)
    , _memory_pool(new memory_pool(program->get_engine(), program->get_config()))
    , _internal(is_internal)
    , _is_primary_stream(is_primary_stream)
    , _enable_profiling(program->get_config().get_enable_profiling())
    , _reset_arguments(true)
    , _shape_predictor(new ShapePredictor(&program->get_engine(), program->get_config().get_shape_predictor_settings())) {
    if (!_internal) {
        net_id = get_unique_net_id();
    }

    calculate_weights_cache_capacity();
    allocate_primitives();
    configure_primitives_second_output();
    build_insts_deps();
    build_exec_order();
    validate_primitives();
    preallocate_shape_info_buffers();
    add_default_output_chains();
    // gtpin integration -- correlation
    init_dispatch_dump();
    // gsoc gtpin
    dump_topology_primitive_map_artifacts();
    // gsoc gtpin
}

network::network(program::ptr program, bool is_internal, bool is_primary_stream)
    :  network(program, program->get_engine().create_stream(program->get_config()), is_internal, is_primary_stream) {}

network::network(engine& engine,
                 const topology& topo,
                 const ExecutionConfig& config,
                 bool is_internal,
                 std::shared_ptr<ov::threading::IStreamsExecutor> task_executor)
    : network(program::build_program(engine, topo, config, task_executor, is_internal), is_internal, true) {}

network::network(engine& engine,
                 const std::set<std::shared_ptr<program_node>>& nodes,
                 const ExecutionConfig& config,
                 std::shared_ptr<ov::threading::IStreamsExecutor> task_executor,
                 bool is_internal)
    : network(program::build_program(engine, nodes, config, task_executor, is_internal), is_internal, true) {}

network::network(program::ptr program, uint16_t stream_id)
    : network(program, program->get_engine().create_stream(program->get_config()), false, stream_id == 0) {}

network::network(program::ptr program, stream::ptr stream, uint16_t stream_id)
    : network(program, stream, false, stream_id == 0) {}

network::~network() {
    if (_program != nullptr)
        _program->cancel_compilation_context();

    // Clear the command queue to prevent errors caused by remaining tasks.
    if (_stream != nullptr)
        _stream->finish();

    _memory_pool->clear_pool_for_network(net_id);
    std::string dump_path = GPU_DEBUG_VALUE_OR(get_config().get_dump_profiling_data_path(), "");

    GPU_DEBUG_IF(!dump_path.empty()) {
        dump_perf_data_raw(dump_path + "/perf_raw" + std::to_string(net_id) + ".csv", false, _exec_order);
    }
    std::string avg_counters_path = GPU_DEBUG_VALUE_OR(get_config().get_average_counters(), "");
    GPU_DEBUG_IF(!avg_counters_path.empty()) {
        dump_average_counters(avg_counters_path, net_id, _exec_order);
    }
}

network::ptr network::allocate_network(stream::ptr stream, program::ptr program, bool is_internal, bool is_primary_stream) {
    return std::make_shared<network>(program, stream, is_internal, is_primary_stream);
}

network::ptr network::allocate_network(engine& engine, program::ptr program, bool is_internal, bool is_primary_stream) {
    auto stream = engine.create_stream(program->get_config());
    return std::make_shared<network>(program, stream, is_internal, is_primary_stream);
}

network::ptr network::build_network(engine& engine,
                                    const topology& topology,
                                    const ExecutionConfig& config,
                                    std::shared_ptr<ov::threading::IStreamsExecutor> task_executor,
                                    bool is_internal) {
    return std::make_shared<network>(engine, topology, config, is_internal, task_executor);
}

network::ptr network::build_network(engine& engine,
                                    const std::set<std::shared_ptr<program_node>>& nodes,
                                    const ExecutionConfig& config,
                                    std::shared_ptr<ov::threading::IStreamsExecutor> task_executor,
                                    bool is_internal) {
    return std::make_shared<network>(engine, nodes, config, task_executor, is_internal);
}

void network::validate_primitives() {
    GPU_DEBUG_DEFINE_MEM_LOGGER("validate_primitives");
    for (auto const& prim : _exec_order) {
        prim->validate();
    }
}

void network::preallocate_shape_info_buffers() {
    GPU_DEBUG_DEFINE_MEM_LOGGER("preallocate_shape_info_buffers");
    int64_t sum = 0;

    /* Use 512 byte alignment for performance */
    const int alignment = 512;

    for (auto const& prim : _exec_order) {
        auto& node = prim->get_node();
        int64_t shape_elements = align_to(node.get_total_shape_info_size(), alignment);
        sum += shape_elements;
    }

    if (sum == 0)
        return;

    auto& engine = get_engine();
    _shape_info_ptr = engine.allocate_memory(layout{{sum}, data_types::i32, format::bfyx}, false);
    size_t offset = 0;
    for (auto const& prim : _exec_order) {
        auto& node = prim->get_node();
        const int64_t shape_elements = node.get_total_shape_info_size();

        if (shape_elements == 0)
            continue;

        auto new_mem = engine.create_subbuffer(*_shape_info_ptr, layout{{shape_elements}, data_types::i32, format::bfyx}, offset);
        prim->set_shape_info_memory(new_mem);

        offset += align_to(shape_elements, alignment) * sizeof(int32_t);
    }
}

void network::set_arguments() {
    if (!_reset_arguments)
        return;

    for (auto const& prim : _exec_order) {
        if (!prim->is_dynamic()) {
            bool can_set_args = true;
            for (auto& dep : prim->dependencies()) {
                // Skip set args for nodes with dynamic & optimized_out dependency
                // This is needed to handle dynamic -> static cases like
                // (dynamic) -> reshape -> (static) -> some_op
                // In that case some_op is static and we may want to set arguments once,
                // but dynamic optimized out reshape means that output buffer of reshape is unavailable
                // and attempt to set args will fail.

                // (dynamic) -> static optimizable reshape -> static optimizable reshape -> some_op
                // In that case, it is a limit about second reshape.
                auto prim = dep.first->get_impl_params()->desc;
                if (dep.first->can_be_optimized() && (dep.first->is_dynamic() ||
                                                      dep.first->output_memory_ptr() == nullptr ||
                                                      prim->type == read_value::type_id()))
                    can_set_args = false;
            }

            if (can_set_args)
                prim->set_arguments();
        }
    }
    _reset_arguments = false;
}

void network::reset_execution(bool wait) {
    if (wait) {
        get_stream().finish();
    }
}

event::ptr network::set_input_data(const primitive_id& id, memory::ptr data, bool need_to_check_memory_to_set) {
    GPU_DEBUG_TRACE_DETAIL << "Set input " << id << " " << data->get_layout().to_short_string() << std::endl;
    auto primitive_inst = find_primitive(id);

    if (primitive_inst->type() != input_layout::type_id()) {
        CLDNN_ERROR_MESSAGE(id, "primitive " + id + " is not an input");
    }
    auto input = std::static_pointer_cast<input_layout_inst>(primitive_inst);
    const bool was_unallocated = !input->output_memory_ptr();
    auto ev = input->set_data(data, need_to_check_memory_to_set);

    if (was_unallocated) {
        // The initial set_arguments() skipped nodes whose dep buffer was null —
        // force a fresh rebind now that the buffer is available.
        _reset_arguments = true;
    }

    // Update the shared mem type hint for the surfaces lock fast-path in execute().
    // We deduplicate by type value to prevent unbounded growth across inferences.
    // Note: this vector is a conservative hint - false positives (stale surface types
    // after input memory switches back to non-surface) are harmless since execute()
    // re-checks live memory state when building in_out_mem.
    // TODO: possibly remove or redesign _in_out_shared_mem_types solution
    if (input->output_memory_ptr()) {
        const auto in_mem_type = input->output_memory_ptr()->get_internal_params().mem_type;
        if (std::find(_in_out_shared_mem_types.begin(), _in_out_shared_mem_types.end(), in_mem_type) == _in_out_shared_mem_types.end())
            _in_out_shared_mem_types.push_back(in_mem_type);
    }

    return ev;
}

void network::add_default_output_chains() {
    GPU_DEBUG_DEFINE_MEM_LOGGER("add_default_output_chains");
    for (auto& output : _outputs) {
        add_output_chain(output);
    }
}

void network::calculate_weights_cache_capacity() {
    auto get_buffer_size = [](const program_node& node) {
        size_t weights_size = 0;
        auto get_size = [](const layout& layout) {
            return layout.is_dynamic() ? 0 : layout.bytes_count();
        };

        #define is_weightable(T) node.is_type<T>() && node.as<T>().weights().is_constant()
        if (node.is_type<data>())
            weights_size = get_size(node.get_output_layout());
        else if (is_weightable(fully_connected))
            weights_size = get_size(node.as<fully_connected>().weights().get_output_layout());
        else if (is_weightable(convolution))
            weights_size = get_size(node.as<convolution>().weights().get_output_layout());
        else if (is_weightable(deconvolution))
            weights_size = get_size(node.as<deconvolution>().weights().get_output_layout());
        #undef is_weightable

        return weights_size;
    };

    size_t total_const_size = 0;
    size_t weights_const_size = 0;
    size_t required_mem_size = 0;
    for (auto node : _program->get_processing_order()) {
        if (node->is_type<fully_connected>() || node->is_type<convolution>() || node->is_type<deconvolution>())
            weights_const_size += get_buffer_size(*node);
        else if (node->is_type<data>())
            total_const_size += get_buffer_size(*node);
    }

    // Sum all weights constants for each stream
    required_mem_size += weights_const_size * get_config().get_num_streams();
    // Add all other constants (shared between streams)
    required_mem_size += total_const_size - weights_const_size;

    if (required_mem_size != 0) {
        const size_t required_weights_cache_capacity = 3;
        const size_t max_device_mem_size = _engine.get_device_info().max_global_mem_size;
        const size_t max_weights_cache_capacity = max_device_mem_size / required_mem_size;

        if (max_weights_cache_capacity > 1)
            _weights_cache_capacity = std::min(max_weights_cache_capacity, required_weights_cache_capacity);
    }
}

network::output_chains_map::iterator network::add_output_chain(std::shared_ptr<primitive_inst>& p_inst) {
    std::vector<primitive_inst*> chain;
    std::stack<const primitive_inst*> candidates;
    auto& eng = get_engine();

    const auto& mem_orig = p_inst->output_memory_ptr();

    auto add_mdata_chain = [&](primitive_inst* p_inst) {
        auto mdata_ptr = dynamic_cast<mutable_data_inst*>(p_inst);
        if (!mdata_ptr)
            return;
        // special handling for mutable data, which can share
        // its attached memory with both its inputs and outputs
        for (auto& dep : p_inst->dependencies()) {
            // check dependencies
            if (dep.first->outputs_allocated() && mem_orig && eng.is_the_same_buffer(*mem_orig, dep.first->output_memory())) {
                chain.push_back(const_cast<primitive_inst*>(dep.first));
            }
            // then second order dependencies
            for (auto& second_dep : dep.first->dependencies()) {
                if (second_dep.first->outputs_allocated() && mem_orig && eng.is_the_same_buffer(*mem_orig, second_dep.first->output_memory())) {
                    chain.push_back(const_cast<primitive_inst*>(second_dep.first));
                }
            }
        }

        //then users
        const auto& user_ids = mdata_ptr->get_user_ids();
        for (const auto& id : user_ids) {
            auto usr_prim = get_primitive(id).get();
            if (usr_prim->outputs_allocated() && mem_orig && eng.is_the_same_buffer(*mem_orig, usr_prim->output_memory())) {
                chain.push_back(usr_prim);
            }
        }
    };

    if (p_inst->can_be_optimized()) {
        candidates.push(p_inst.get());
    } else {
        chain.push_back(p_inst.get());
    }
    add_mdata_chain(p_inst.get());

    // find all dependencies that are 'optimized'
    while (!candidates.empty()) {
        auto cand = candidates.top();
        candidates.pop();
        // Add cand inst to the chain when cand's output is not allocated yet.
        if (!p_inst->outputs_allocated()
            || (cand->outputs_allocated() && eng.is_the_same_buffer(*mem_orig, cand->output_memory()))) {
            auto nc_cand = const_cast<primitive_inst*>(cand);
            chain.push_back(nc_cand);
            add_mdata_chain(nc_cand);
        }

        for (auto& dep : cand->dependencies()) {
            if (dep.first->can_be_optimized()) {
                candidates.push(dep.first);
            } else {
                if (dep.first->outputs_allocated()) {
                    const auto& mem_dep = dep.first->output_memory();
                    // Add dep inst to the chain when dep's output is not allocated yet.
                    if (!p_inst->outputs_allocated()
                        || eng.is_the_same_buffer(*mem_orig, mem_dep)) {
                        auto nc_dep = const_cast<primitive_inst*>(dep.first);
                        chain.push_back(nc_dep);
                        add_mdata_chain(nc_dep);
                    }
                }
            }
        }
    }

    std::sort(chain.begin(), chain.end());
    chain.erase(std::unique(chain.begin(), chain.end()), chain.end());
    return _output_chains.insert({ p_inst->id(), chain }).first;
}

std::vector<event::ptr> network::set_output_memory(const primitive_id& id, memory::ptr mem_new, bool is_remote) {
    GPU_DEBUG_TRACE_DETAIL << "Set output " << id << " " << mem_new->get_layout().to_short_string() << std::endl;
    std::vector<event::ptr> ret_ev;
    std::shared_ptr<primitive_inst> p_inst = find_primitive(id);

    auto iter = std::find(_outputs.begin(), _outputs.end(), p_inst);
    if (iter == _outputs.end())
        throw std::runtime_error("primitive: " + id + " is not a network output");

    if (is_remote) {
        _output_remote_mem_ptrs[id] = mem_new;
    }

    auto& eng = get_engine();
    // locate primitive chain for this output
    // if no chain found - add it
    auto o_iter = _output_chains.find(id);
    if (o_iter == _output_chains.end()) {
        o_iter = add_output_chain(p_inst);
    }

    for (auto& prim : o_iter->second) {
        auto mem = mem_new;
        if (!prim->is_dynamic() && mem_new && prim->output_memory_ptr())
            mem = eng.reinterpret_buffer(*mem_new, prim->output_memory().get_layout());

        ret_ev.push_back(prim->set_output_memory(mem, (!prim->is_dynamic() || !is_remote)));
        if (!_reset_arguments &&
            (prim->type() != cldnn::data::type_id() && (prim->type() != cldnn::mutable_data::type_id() || !prim->dependencies().empty()))) {
            prim->set_arguments();
        }
    }
    return ret_ev;
}

std::shared_ptr<primitive_inst> cldnn::network::find_primitive(const primitive_id& id) const {
    auto it = _primitives.find(id);
    OPENVINO_ASSERT(it != _primitives.end(), "[GPU] Network doesn't contain primitive ", id);
    return it->second;
}

std::string network::get_primitive_info(const primitive_id& id) const {
    const auto& node = _program->get_node(id);
    return node.type()->to_string(node);
}

bool network::does_node_need_lockable_output(const primitive_id& id) const {
    auto prim_inst = find_primitive(id);

    const auto& node = prim_inst->get_node();
    if (node.is_type<input_layout>()) {
        for (const auto& user : node.get_users()) {
            const auto& lockable_input_ids = user->get_lockable_input_ids();
            if (lockable_input_ids.count(user->get_dependency_index(node)) != 0u) {
                return true;
            }
        }

        return false;
    } else {
        return prim_inst->get_impl() ? prim_inst->get_impl()->is_cpu() : true;
    }
}

std::string network::get_implementation_info(const primitive_id& id) const {
    try {
        auto it = _primitives.find(id);
        if (it != _primitives.end()) {
            auto* impl = it->second->get_impl();
            auto kernel_name = impl ? impl->get_kernel_name() : "";
            if (!kernel_name.empty()) {
                if (_program != nullptr) {
                    const auto& node = it->second->get_node();
                    return kernel_name + "__" + dt_to_str(_program->get_inference_precision(node));
                } else {
                    return kernel_name;
                }
            }
        }
    } catch (...) { }

    if (_program == nullptr)
        return "undef";

    return _program->get_implementation_info(id);
}

memory::ptr network::get_output_memory(const primitive_id& output_id) {
    return get_primitive(output_id)->output_memory_ptr();
}

layout network::get_output_layout(const primitive_id& output_id) const {
    return get_primitive(output_id)->get_output_layout();
}

void network::allocate_primitives() {
    GPU_DEBUG_DEFINE_MEM_LOGGER("allocate_primitives");
    const auto& ao = _program->get_allocating_order();
    for (auto& node_id : ao) {
        allocate_primitive_instance(_program->get_node(node_id));
    }

    auto& po = _program->get_processing_order();

    // Update the output memory address of optimized-out layer if it is not valid.
    for (auto const& node : po) {
        if (node->can_be_optimized() && !node->is_dynamic() &&
            (node->get_dependencies().empty() || !node->get_dependency(0).is_type<read_value>())) {
            auto opt_inst = _primitives.at(node->id());
            // build deps when prim_inst does not update dependencies yet.
            if (!node->get_dependencies().empty() && opt_inst->dependencies().empty()) {
                opt_inst->build_deps();
            }
            // Skip if the dependency's memory is not yet allocated (e.g. lazy input_layout).
            // The output memory will be set up at runtime when the input becomes available.
            if (!opt_inst->dependencies().empty() && opt_inst->dep_memory_ptr(0) == nullptr)
                continue;
            opt_inst->update_output_memory();
        }
    }

    // allocate intermediate buffers
    for (auto const& node : po) {
        auto prim = _primitives[node->id()];
        prim->allocate_internal_buffers();
    }
}

void network::configure_primitives_second_output() {
    GPU_DEBUG_DEFINE_MEM_LOGGER("configure_primitives_second_output");
    std::map<cldnn::memory::ptr, std::vector<const cldnn::program_node*>> mutable_datas_ptrs;
    for (auto& inst : _primitives) {
        auto& node = inst.second->get_node();

        if (!node.is_type<mutable_data>())
            continue;

        mutable_datas_ptrs[node.as<mutable_data>().get_attached_memory_ptr()].push_back(&node);
    }

    for (auto item : mutable_datas_ptrs) {
        if (item.second.size() != 2)
            continue;

        auto is_first_node_input_md = [&](const cldnn::program_node* first,
                                          const cldnn::program_node* second) {
            for (auto user : first->get_users()) {
                for (auto next_user : user->get_users()) {
                    if (next_user == second)
                        return true;
                }
            }
            return false;
        };

        auto is_first_node_input = is_first_node_input_md(item.second[0], item.second[1]);

        auto input_md_inst = is_first_node_input ? _primitives[item.second[0]->id()] : _primitives[item.second[1]->id()];
        auto output_md_inst = is_first_node_input ? _primitives[item.second[1]->id()] : _primitives[item.second[0]->id()];

        output_md_inst->set_output_memory(input_md_inst->output_memory_ptr(), false);
    }
}

void network::build_insts_deps() {
    GPU_DEBUG_DEFINE_MEM_LOGGER("build_insts_deps");
    for (auto& inst : _primitives) {
        inst.second->build_deps();
        inst.second->init_users();
        inst.second->configure_shape_of_dependencies();
    }
}

void network::build_exec_order() {
    GPU_DEBUG_DEFINE_MEM_LOGGER("build_exec_order");
    if (!_is_dynamic) {
        for (auto& node : _program->get_processing_order()) {
            if (!node->is_type<data>() && (!node->is_type<mutable_data>() || !node->get_dependencies().empty())) {
                add_to_exec_order(node->id());
            }
        }
    } else {
        auto is_runtime_optimized_concat = [&](const program_node* node) {
            return (node->is_dynamic() && node->is_type<concatenation>() && node->can_be_optimized());
        };
        auto is_allowed_pred_for_runtime_optimized_concat = [&](const program_node* node) {
            return (!node->is_type<data>() && (!node->is_type<mutable_data>() || !node->get_dependencies().empty()) &&
                    node->get_users().size() == 1 && is_runtime_optimized_concat(node->get_users().front()));
        };
        for (auto& node : _program->get_processing_order()) {
            if (!node->is_type<data>() && (!node->is_type<mutable_data>() || !node->get_dependencies().empty())) {
                if (is_allowed_pred_for_runtime_optimized_concat(node)) {
                    continue;
                } else if (is_runtime_optimized_concat(node)) {
                    // For in-place concat applied at runtime, we need to do update_shape for all other predecessors of the concat user.
                    // i.e., We need to make sure that all the preds of them are already updated too.
                    for (auto dep : node->get_dependencies()) {
                        if (!dep.first->is_type<data>()) {
                            add_to_exec_order(dep.first->id());
                        }
                    }
                }
                add_to_exec_order(node->id());
            }
        }
    }
}

bool network::contains_state(const std::string& variable_id) {
    auto it = _state_initializers.find(variable_id);
    return it != _state_initializers.end();
}

memory& network::get_output_remote_memory(const primitive_id& id) const {
    OPENVINO_ASSERT(_output_remote_mem_ptrs.count(id) == 1, "[GPU] Can't get output remote memory with ", id);
    return *_output_remote_mem_ptrs.at(id);
}

bool network::has_output_remote_memory_ptr(const primitive_id& id) const {
    auto it = _output_remote_mem_ptrs.find(id);
    return it != _output_remote_mem_ptrs.end();
}

void network::reset_output_remote_memory_ptrs() {
    if (!_output_remote_mem_ptrs.empty()) {
        _output_remote_mem_ptrs.clear();
    }
}

void network::invalidate_output_memory_chain(const primitive_id& id) {
    auto p_inst = find_primitive(id);
    p_inst->clear_output_memory();

    auto o_iter = _output_chains.find(id);
    if (o_iter != _output_chains.end()) {
        for (auto* prim : o_iter->second) {
            if (prim != p_inst.get()) {
                prim->clear_output_memory();
            }
        }
    }
}

void network::invalidate_ext_block_compute_nodes(const primitive_id& output_id) {
    // Walk backward from the output node through optimized single-dependency
    // predecessors until we reach the compute node (the first non-optimized one).
    // Clear its _outputs[0] so that prepare_primitive's null-check triggers
    // realloc_if_needed, which will re-probe forward and pick up the new ext_block
    // buffer after a double-buffer flip.
    //
    // Stop at runtime-skippable nodes: they may have had can_be_optimized flipped
    // to false by do_runtime_skip_*() and should not participate in the ext_block chain.
    auto cursor = find_primitive(output_id);
    while (cursor->can_be_optimized() && !cursor->dependencies().empty()) {
        cursor->clear_output_memory();
        GPU_DEBUG_TRACE_DETAIL << "[double-buffer] cleared output memory on optimized node " << cursor->id() << std::endl;
        auto dep_id = cursor->dependencies().front().first->id();
        auto dep = find_primitive(dep_id);
        // Stop before runtime-skippable nodes: their can_be_optimized() may
        // have been re-evaluated at runtime — they are the compute boundary.
        if (dep->get_node().is_runtime_skippable())
            break;
        cursor = dep;
    }
    // cursor is now the compute node — clear its output so it re-acquires from ext_block
    if (!cursor->has_inner_networks() && !cursor->can_be_optimized()) {
        cursor->clear_output_memory();
        GPU_DEBUG_TRACE_DETAIL << "[double-buffer] cleared output memory on compute node " << cursor->id() << std::endl;
    }
}

void network::register_output_memory_block(const primitive_id& id, ov::intel_gpu::OutputMemoryBlock* block) {
    OPENVINO_ASSERT(block != nullptr, "[GPU] Use unregister path (nullptr) via clear_output_memory_blocks or erase");

    auto [it, inserted] = _output_memory_blocks.emplace(id, block);
    if (!inserted) {
        if (it->second == block)
            return;  // Same block already registered — nothing to do
        it->second = block;
    }
}

void network::unregister_output_memory_block(const primitive_id& id) {
    auto it = _output_memory_blocks.find(id);
    if (it != _output_memory_blocks.end()) {
        _output_memory_blocks.erase(it);
        invalidate_ext_block_compute_nodes(id);
    }
}

ov::intel_gpu::OutputMemoryBlock* network::get_output_memory_block(const primitive_id& id) const {
    auto it = _output_memory_blocks.find(id);
    return (it != _output_memory_blocks.end()) ? it->second : nullptr;
}

void network::clear_output_memory_blocks() {
    for (auto& [prim_id, block_ptr] : _output_memory_blocks) {
        invalidate_ext_block_compute_nodes(prim_id);
    }
    _output_memory_blocks.clear();
}

void network::add_to_exec_order(const primitive_id& id) {
    auto inst = get_primitive(id);
    _exec_order.push_back(inst);
}

std::map<primitive_id, network_output> network::execute(const std::vector<event::ptr>& dependencies) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "NetworkImpl::Execute");
    NETWORK_DEBUG(*this);

    // Wait for previous execution completion
    reset_execution(false);

    std::vector<memory::ptr> in_out_mem;
    bool shared_mem_found = std::any_of(_in_out_shared_mem_types.begin(),
                                        _in_out_shared_mem_types.end(),
                                        surfaces_lock::is_lock_needed);

    if (shared_mem_found) {
        for (auto& inst : _inputs) {
            if (inst->output_memory_ptr() &&
                surfaces_lock::is_lock_needed(inst->output_memory_ptr()->get_internal_params().mem_type))
                in_out_mem.push_back(inst->output_memory_ptr());
        }

        for (auto& inst : _outputs) {
            if (inst->output_memory_ptr() &&
                surfaces_lock::is_lock_needed(inst->output_memory_ptr()->get_internal_params().mem_type))
                in_out_mem.push_back(inst->output_memory_ptr());
        }
    }

    for (auto& inst : _read_values) {
        const auto& prim = inst->get_node().as<read_value>().get_primitive();
        auto it = _state_initializers.find(prim->variable_id);
        if (it != _state_initializers.end()) {
            const auto& variable = get_variable(prim->variable_id);
            if (variable.is_set()) {
                for (auto& init_inst : it->second) {
                    init_inst->set_flag(ExecutionFlags::SKIP);
                }
            }
        }
    }

    // We shouldn't call create_surfaces_lock function constantly here, but due to
    // some changes in assembler code, performance drops in case if we move it under
    // `shared_mem_found` condition (it somehow connected with get_cl_queue() - this function call
    // makes asm faster for some reasons). So, as WA we keep this create_surfaces_lock here
    // with empty memory vector and do nothing inside this function for saving performance
    // in some cases.
    auto surf_lock = get_stream().create_surfaces_lock(in_out_mem);

    execute_impl(dependencies);

    std::map<primitive_id, network_output> result;
    for (auto& inst : _outputs) {
        event::ptr ev = nullptr;
        const auto& id = inst->id();
        if (get_stream().get_queue_type() == QueueTypes::out_of_order || _enable_profiling)
            ev = inst->get_impl_params()->out_event;

        result.emplace(id, network_output(ev, inst->output_memory_ptr(0), get_stream_ptr(), inst->get_output_layout(0)));
    }

    return result;
}

const event::ptr& network::get_primitive_event(const primitive_id& id) const {
    return get_primitive(id)->get_impl_params()->out_event;
}

bool network::has_event(const primitive_id& id) const {
    auto it = _primitives.find(id);
    if (it == _primitives.end())
        return false;

    return it->second->get_impl_params()->out_event != nullptr;
}

void network::execute_impl(const std::vector<event::ptr>& events) {
    set_arguments();

    // This extra flush command is needed for dynamic models in both cases of out_of_order / in_order operating mode
    // since it reduces `bubbles` number in pipeline and GPU's idle time by timely flushing new kernels to device.
    // The freqency of flushing (16) is selected empirically, see details in tickets 116365, 116287, 139931.
    const bool needs_flushing = _is_dynamic;
    const size_t flush_frequency = needs_flushing ? 16 : 0;
    size_t executed_prims = 0;
#ifdef GPU_DEBUG_CONFIG
    // gtpin integration -- correlation
    _dispatch_index = 0;
#endif

    for (auto& inst : _exec_order) {
        NODE_DEBUG(*inst);
        OV_ITT_SCOPED_TASK_BASE(ov::intel_gpu::itt::domains::intel_gpu_op, openvino::itt::handle(inst->id()));

        inst->reset_events();

        if (inst->is_input()) {
            inst->add_dep_events(events);
        }

        inst->prepare_primitive();
        inst->execute();

        executed_prims++;
        if (needs_flushing && executed_prims % flush_frequency == 0)
            get_stream().flush();
    }

    // Using output of previous network as input to another one may cause hazard (in OOOQ mode) if user would not
    // provide proper event to execution. Flushing pipeline should prevent this kind of issues.
    // In scenarios with a big number of very small networks it can provide performance drop.
    get_stream().flush();

    // Reset all flags for the next execution
    for (auto& inst : _exec_order) {
        inst->reset_flags();
    }
}

void network::init_dispatch_dump() {
#ifdef GPU_DEBUG_CONFIG
    const std::string dump_path = GPU_DEBUG_VALUE_OR(get_config().get_dump_dispatch_map_path(), "");
    if (dump_path.empty()) {
        return;
    }

    std::error_code error_code;
    std::filesystem::create_directories(dump_path, error_code);
    if (error_code) {
        GPU_DEBUG_INFO << "[dispatch_map] Failed to create dump directory " << dump_path
                       << ". error=" << error_code.message() << std::endl;
        return;
    }

    _dispatch_dump_file_path = dump_path + "/dispatch_map.csv";

    std::lock_guard<std::mutex> lock(get_dispatch_dump_mutex());
    bool write_header = false;
    {
        std::ifstream existing_file(_dispatch_dump_file_path, std::ios::binary | std::ios::ate);
        write_header = !existing_file.good() || existing_file.tellg() == 0;
    }

    _dispatch_dump_stream.open(_dispatch_dump_file_path, std::ios::out | std::ios::app);
    if (!_dispatch_dump_stream.is_open()) {
        GPU_DEBUG_INFO << "[dispatch_map] Failed to open dump file " << _dispatch_dump_file_path << std::endl;
        return;
    }

    if (write_header) {
        _dispatch_dump_stream << "net_id,iteration,dispatch_index,global_dispatch_id,primitive_id,primitive_type,implementation,kernel_index,kernel_entry,batch_hash,input_arg_addresses,output_arg_addresses,output_memory_addresses\n";
        _dispatch_dump_stream.flush();
    }
#endif
}

// gsoc gtpin
void network::dump_topology_primitive_map_artifacts() const {
#ifdef GPU_DEBUG_CONFIG
    const std::string dump_path = GPU_DEBUG_VALUE_OR(get_config().get_dump_topology_primitive_map_path(), "");
    if (dump_path.empty() || _program == nullptr || _internal) {
        return;
    }

    const auto detail_path = dump_path + "/ov_topdown_primitive_rows" + std::to_string(net_id) + ".csv";
    const auto summary_path = dump_path + "/ov_topdown_primitive_summary" + std::to_string(net_id) + ".csv";

    std::map<std::string, primitive_info> primitive_info_by_id;
    for (const auto& info : get_primitives_info()) {
        primitive_info_by_id.emplace(info.original_id, info);
    }

    std::vector<topdown_primitive_row> rows;
    std::map<std::string, topdown_summary_row> summary_by_origin;
    rows.reserve(_primitives.size());

    for (const auto& primitive_entry : _primitives) {
        const auto& primitive_id = primitive_entry.first;
        const auto& inst = primitive_entry.second;
        const auto prim = inst->get_node().get_primitive();

        topdown_primitive_row row;
        row.origin_op_name = prim->origin_op_name.empty() ? inst->org_id() : prim->origin_op_name;
        row.primitive_id = primitive_id;
        row.original_primitive_id = inst->org_id();
        row.primitive_type = inst->desc()->type_string();
        row.implementation = get_implementation_info(primitive_id);
        row.is_input = std::find_if(_inputs.begin(), _inputs.end(), [&](const std::shared_ptr<primitive_inst>& input) {
            return input->id() == primitive_id;
        }) != _inputs.end();
        row.is_output = std::find_if(_outputs.begin(), _outputs.end(), [&](const std::shared_ptr<primitive_inst>& output) {
            return output->id() == primitive_id;
        }) != _outputs.end();

        const auto info_it = primitive_info_by_id.find(primitive_id);
        if (info_it != primitive_info_by_id.end()) {
            row.exec_id = info_it->second.exec_id;
            row.dependencies.assign(info_it->second.c_dependencies.begin(), info_it->second.c_dependencies.end());
            row.users.assign(info_it->second.c_users.begin(), info_it->second.c_users.end());
            row.fused_ids.assign(info_it->second.c_fused_ids.begin(), info_it->second.c_fused_ids.end());
        }

        auto& summary = summary_by_origin[row.origin_op_name];
        summary.primitive_ids.insert(row.primitive_id);
        summary.original_primitive_ids.insert(row.original_primitive_id);
        summary.primitive_types.insert(row.primitive_type);
        summary.implementations.insert(row.implementation);

        rows.push_back(std::move(row));
    }

    for (const auto& optimized_out_id : _program->get_optimized_out()) {
        std::string origin_name = optimized_out_id;
        const auto separator_pos = optimized_out_id.find(':');
        if (separator_pos != std::string::npos && separator_pos + 1 < optimized_out_id.size()) {
            origin_name = optimized_out_id.substr(separator_pos + 1);
        }
        summary_by_origin[origin_name].optimized_out_ids.insert(optimized_out_id);
    }

    std::sort(rows.begin(), rows.end(), [](const topdown_primitive_row& lhs, const topdown_primitive_row& rhs) {
        if (lhs.origin_op_name != rhs.origin_op_name) {
            return lhs.origin_op_name < rhs.origin_op_name;
        }
        if (lhs.exec_id != rhs.exec_id) {
            return lhs.exec_id < rhs.exec_id;
        }
        return lhs.primitive_id < rhs.primitive_id;
    });

    std::ofstream detail_file(detail_path, std::ios::out | std::ios::trunc);
    if (detail_file.is_open()) {
        detail_file << "net_id,origin_op_name,primitive_id,original_primitive_id,primitive_type,implementation,exec_id,is_input,is_output,dependencies,users,fused_ids\n";
        for (const auto& row : rows) {
            detail_file << net_id << ","
                        << csv_escape(row.origin_op_name) << ","
                        << csv_escape(row.primitive_id) << ","
                        << csv_escape(row.original_primitive_id) << ","
                        << csv_escape(row.primitive_type) << ","
                        << csv_escape(row.implementation) << ","
                        << row.exec_id << ","
                        << (row.is_input ? "true" : "false") << ","
                        << (row.is_output ? "true" : "false") << ","
                        << csv_escape(join_strings(row.dependencies, ";")) << ","
                        << csv_escape(join_strings(row.users, ";")) << ","
                        << csv_escape(join_strings(row.fused_ids, ";")) << "\n";
        }
    }

    std::ofstream summary_file(summary_path, std::ios::out | std::ios::trunc);
    if (summary_file.is_open()) {
        summary_file << "net_id,origin_op_name,spawned_primitive_count,primitive_ids,original_primitive_ids,primitive_types,implementations,optimized_out_ids\n";
        for (const auto& entry : summary_by_origin) {
            summary_file << net_id << ","
                         << csv_escape(entry.first) << ","
                         << entry.second.primitive_ids.size() << ","
                         << csv_escape(join_strings(entry.second.primitive_ids, ";")) << ","
                         << csv_escape(join_strings(entry.second.original_primitive_ids, ";")) << ","
                         << csv_escape(join_strings(entry.second.primitive_types, ";")) << ","
                         << csv_escape(join_strings(entry.second.implementations, ";")) << ","
                         << csv_escape(join_strings(entry.second.optimized_out_ids, ";")) << "\n";
        }
    }
#endif
}

void network::dump_topology_primitive_map() const {
    dump_topology_primitive_map_artifacts();
}
// gsoc gtpin

// gsoc gtpin start
void network::dump_dispatch_row(const primitive_inst& inst,
                                size_t kernel_index,
                                const std::string& kernel_entry_override,
                                const kernel_arguments_desc& args_desc,
                                const kernel_arguments_data& args) {
#ifdef GPU_DEBUG_CONFIG
    if (!_dispatch_dump_stream.is_open()) {
        return;
    }

    std::string implementation;
    std::string kernel_entry;
    std::string batch_hash;
    std::string metadata_source = "none";
    // gsoc project: current OV kernel dump metadata API requires impl params and
    // returns KernelDumpInfo rather than the older pair<string, string> form.
    const auto* impl_params = inst.get_impl_params();
    if (const auto* impl = inst.get_impl()) {
        implementation = impl->get_kernel_name();
        const auto kernel_dump_info = impl_params ? impl->get_kernels_dump_info(*impl_params) : KernelDumpInfo{};
        batch_hash = kernel_dump_info.get_batch_hash();
        if (!kernel_entry_override.empty()) {
            kernel_entry = kernel_entry_override;
            metadata_source = "per_dispatch_override";
        } else {
            const auto kernel_entries = split_kernel_entries(kernel_dump_info.get_entries());
            if (kernel_index < kernel_entries.size()) {
                kernel_entry = kernel_entries[kernel_index];
                metadata_source = "runtime_impl_indexed";
            } else {
                kernel_entry = kernel_dump_info.get_entries();
                if (!kernel_entry.empty() || !batch_hash.empty()) {
                    metadata_source = "runtime_impl_raw";
                }
            }
        }
        if ((!kernel_entry.empty() || !batch_hash.empty()) && metadata_source == "none") {
            metadata_source = "runtime_impl";
        }
    }

    if (kernel_entry.empty() && inst.has_node()) {
        if (const auto* selected_impl = inst.get_node().get_selected_impl()) {
            const auto kernel_dump_info = impl_params ? selected_impl->get_kernels_dump_info(*impl_params) : KernelDumpInfo{};
            if (batch_hash.empty()) {
                batch_hash = kernel_dump_info.get_batch_hash();
            }
            if (!kernel_entry_override.empty()) {
                kernel_entry = kernel_entry_override;
            } else {
                const auto kernel_entries = split_kernel_entries(kernel_dump_info.get_entries());
                kernel_entry = kernel_index < kernel_entries.size() ? kernel_entries[kernel_index] : kernel_dump_info.get_entries();
            }
            if (!kernel_entry.empty() || !batch_hash.empty()) {
                metadata_source = "selected_impl_fallback";
                GPU_DEBUG_INFO << "[dispatch_map] Fallback kernel metadata for primitive " << inst.id()
                               << " from selected_impl. impl=" << implementation
                               << " kernel_entry=" << kernel_entry
                               << " batch_hash=" << batch_hash << std::endl;
            }
        }
    }

    if (kernel_entry.empty()) {
        GPU_DEBUG_INFO << "[dispatch_map] Missing kernel_entry for primitive " << inst.id()
                       << " impl=" << implementation
                       << " batch_hash=" << batch_hash
                       << " source=" << metadata_source << std::endl;
    }

    const auto input_arg_addresses = get_kernel_input_arg_addresses(args_desc, args);
    const auto output_arg_addresses = get_kernel_output_arg_addresses(args_desc, args);
    const auto output_memory_addresses = get_output_memory_addresses(args);
    const auto local_dispatch_index = _dispatch_index++;
    const auto global_dispatch_id = acquire_global_dispatch_dump_id();

    std::lock_guard<std::mutex> lock(get_dispatch_dump_mutex());
    _dispatch_dump_stream << net_id << ","
                          << get_current_iteration_num() << ","
                          << local_dispatch_index << ","
                          << global_dispatch_id << ","
                          << csv_escape(inst.id()) << ","
                          << csv_escape(inst.desc()->type_string()) << ","
                          << csv_escape(implementation) << ","
                          << kernel_index << ","
                          << csv_escape(kernel_entry) << ","
                          << csv_escape(batch_hash) << ","
                          << csv_escape(input_arg_addresses) << ","
                          << csv_escape(output_arg_addresses) << ","
                          << csv_escape(output_memory_addresses) << "\n";
    _dispatch_dump_stream.flush();
#else
    (void)inst;
    (void)kernel_index;
    (void)kernel_entry_override;
    (void)args_desc;
    (void)args;
#endif
}
// gsoc gtpin end

std::vector<primitive_id> network::get_input_ids() const {
    std::vector<primitive_id> ret;
    ret.reserve(_inputs.size());
    for (auto const& input : _inputs) ret.push_back(input->id());
    return ret;
}

std::vector<layout> network::get_input_layouts() const {
    std::vector<layout> ret;
    ret.reserve(_inputs.size());
    for (auto const& input : _inputs)
        ret.push_back(input->output_memory_ptr() ? input->output_memory_ptr()->get_layout() : input->get_output_layout());
    return ret;
}

std::vector<primitive_id> network::get_output_ids() const {
    std::vector<primitive_id> ret;
    ret.reserve(_outputs.size());
    for (auto const& output : _outputs) ret.push_back(output->id());
    return ret;
}

std::vector<primitive_id> network::get_executed_primitive_ids() const {
    std::vector<primitive_id> ret;
    ret.reserve(_exec_order.size());
    for (auto const& executed_primitive : _exec_order) {
        ret.push_back(executed_primitive->id());
    }
    return ret;
}

std::vector<primitive_id> network::get_all_primitive_ids() const {
    std::vector<primitive_id> ret;
    ret.reserve(_primitives.size());
    for (auto const& primitive : _primitives)
        if (primitive.second->can_be_optimized())
            ret.push_back("_optimized_");
        else
            ret.push_back(primitive.second->id());
    return ret;
}

std::vector<primitive_id> network::get_all_primitive_org_ids() const {
    std::vector<primitive_id> ret;
    ret.reserve(_primitives.size());
    for (auto const& primitive : _primitives) ret.push_back(primitive.second->org_id());
    return ret;
}

const program::primitives_info& network::get_primitives_info() const {
    return (_program == nullptr) ? _prims_info : _program->get_primitives_info();
}

const program::graph_optimizer_info& network::get_optimizer_passes_info() const {
    return _program->get_optimizer_passes_info();
}

std::map<primitive_id, primitive_id> network::get_ext_id_mapping() const {
    std::map<primitive_id, primitive_id> result;
    for (auto& prim : _primitives) {
        result.emplace(prim.first, prim.second->get_node().get_primitive()->origin_op_name);
    }
    for (auto& opt_id : _program->get_optimized_out()) {
        std::string ext_id = opt_id;
        if (opt_id.find(":") != std::string::npos) {
            ext_id = opt_id.substr(opt_id.find(":") + 1, opt_id.length());
        }
        result.emplace(opt_id, ext_id);
    }
    return result;
}

std::shared_ptr<primitive_inst> network::get_primitive(const primitive_id& id) {
    if (!_primitives.count(id))
        allocate_primitive_instance(_program->get_node(id));

    return _primitives.at(id);
}

std::shared_ptr<const primitive_inst> network::get_primitive(const primitive_id& id) const {
    OPENVINO_ASSERT(_primitives.count(id) == 1, "[GPU] Can't get primitive with ", id, " id: primitive with such name hasn't been found in processing order");
    return _primitives.at(id);
}

std::vector<primitive_inst*> network::get_primitives(const std::vector<primitive_id>& ids) {
    std::vector<primitive_inst*> result(ids.size());
    std::transform(std::begin(ids), std::end(ids), std::begin(result), [&](const primitive_id& id) {
        return get_primitive(id).get();
    });
    return result;
}

std::vector<std::pair<primitive_inst*, int>> network::get_primitives(const std::vector<std::pair<program_node*, int>>& nodes) {
    std::vector<std::pair<primitive_inst*, int>> result(nodes.size());
    std::transform(std::begin(nodes), std::end(nodes), std::begin(result), [&](const std::pair<program_node*, int>& node) {
        return std::make_pair(get_primitive(node.first->id()).get(), node.second);
    });
    return result;
}

void network::allocate_primitive_instance(program_node const& node) {
    if (_primitives.count(node.id()))
        return;

    GPU_DEBUG_TRACE_DETAIL << node.id() << ": allocate primitive instance" << std::endl;

    auto inst = node.type()->create_instance(*this, node);

    std::function<bool(const program_node&)> is_mutable_input = [&is_mutable_input](const program_node& node) {
        for (auto& dep : node.get_dependencies()) {
            const auto dep_node = dep.first;
            if (dep_node->is_type<input_layout>() || dep_node->is_type<mutable_data>() || (dep_node->is_type<read_value>() && !dep_node->can_be_optimized())) {
                return true;
            }
            if (dep_node->can_be_optimized()) {
                if (is_mutable_input(*dep_node)) {
                    return true;
                }
            }
        }
        return false;
    };

    if (is_mutable_input(node)) {
        inst->set_mutable_input(true);
    }

    if (inst->is_dynamic()) {
        _is_dynamic = true;
    }

    if (!node.is_type<data>()) {
        inst->set_flag(ExecutionFlags::IMPL_CHANGED);
        inst->set_flag(ExecutionFlags::SHAPE_CHANGED);
        inst->set_flag(ExecutionFlags::MEMORY_CHANGED);
    }


    _primitives[node.id()] = inst;
    if (node.is_type<input_layout>()) {
        if (inst->output_memory_ptr())
            _in_out_shared_mem_types.push_back(inst->output_memory_ptr()->get_internal_params().mem_type);
        _inputs.push_back(inst);
    }

    if (node.is_output()) {
        if (inst->output_memory_ptr())
            _in_out_shared_mem_types.push_back(inst->output_memory_ptr()->get_internal_params().mem_type);
        _outputs.push_back(inst);
        if (node.is_type<data>())
            _data_outputs.push_back(inst);
    }

    bool is_lora_state = false;
    if (node.is_type<read_value>()) {
        _read_values.push_back(inst);
        const auto& variable_id = node.as<read_value>().get_primitive()->variable_id;
        if (_program->contains_state(variable_id)) {
            for (const auto& id : _program->get_initializers(variable_id)) {
                _state_initializers[variable_id].push_back(get_primitive(id));
            }
        }
        const auto& users = node.get_users();
        if (!users.empty()) {
            is_lora_state = users.front()->is_type<lora>();
        }
    }

    if (auto state_prim = std::dynamic_pointer_cast<memory_state::variable>(inst)) {
        auto prim = inst->get_node().get_primitive();

        bool transpose_required = false;
        if (is_lora_state) {
            const auto& lora_prim = node.get_users().front()->as<lora>().get_primitive();
            for (size_t state_idx : {2, 4, 5, 7, 8, 10}) {
                if (state_idx < lora_prim->input.size() &&
                    lora_prim->input[state_idx].pid == node.id()) {
                    transpose_required = lora_prim->transposed_states;
                }
            }
        }
        set_variables_state_info(state_prim->variable_id(),
                                 node.get_output_layout(0),
                                 state_prim->get_user_specified_type(),
                                 prim.get(),
                                 std::dynamic_pointer_cast<memory_state::releasable_variable>(inst),
                                 transpose_required);
    }

    if (node.is_constant()) {
        transfer_memory_to_device(inst, node);
    }
}

void network::transfer_memory_to_device(std::shared_ptr<primitive_inst> instance, program_node const& node) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "NetworkImpl::TransferMemory");
    auto& inst_mem = instance->output_memory();
    auto alloc_type = inst_mem.get_allocation_type();

    const auto& users = node.get_users();
    if (users.size() == 1
        && users.front()->is_type<reshape>()
        && users.front()->is_dynamic())
            return;
    if (node.is_type<data>() && node.as<data>().get_primitive()->skip_device_transfer()) {
        return;
    }
    // Do not transfer memory if a user requires lockable memory.
    // If memory is used in both gpu and cpu implementations, primitive itself is responsible for correct allocation type
    if (node.need_lockable_memory())
        return;

    if (!get_engine().supports_allocation(allocation_type::usm_device))
        return;

    if (!get_engine().get_device_info().has_separate_cache)
        return;

    if (node.is_shape_infer_dep())
        return;

    if (inst_mem.count() == 0)
        return;

    if (alloc_type == allocation_type::usm_host || alloc_type == allocation_type::usm_shared) {
        // usm_device memory does not provide performance benefits on the integrated Xe2+ platforms
        if (get_engine().get_device_info().arch >= gpu_arch::xe2 &&
            get_engine().get_device_info().dev_type == device_type::integrated_gpu) {
            return;
        }

        // Allocate and transfer memory
        auto device_mem = inst_mem.get_engine()->allocate_memory(inst_mem.get_layout(), allocation_type::usm_device, false);
        device_mem->copy_from(get_stream(), inst_mem);
        GPU_DEBUG_LOG << "[" << node.id() << ": constant]" << std::endl;
        _memory_pool->release_memory(&inst_mem, node.get_unique_id(), node.id(), get_id());
        instance->set_output_memory(device_mem);
    }
}

void network::set_variable(const std::string& name, const std::shared_ptr<ov::intel_gpu::VariableStateBase>& variable) {
    GPU_DEBUG_TRACE_DETAIL << "Set variable " << name << " " << variable->get_layout().to_short_string() << std::endl;
    _variables_states[name] = variable;
}

bool network::has_variable(const std::string &variable_id) const {
    return _variables_states.find(variable_id) != _variables_states.end();
}

ov::intel_gpu::VariableStateBase& network::get_variable(const std::string &variable_id) const {
    auto it = _variables_states.find(variable_id);
    OPENVINO_ASSERT(it != _variables_states.end(), "[GPU] ", variable_id, " variable not found");
    return *it->second;
}

const ov::intel_gpu::VariableStateInfo& network::get_variable_info(const std::string &variable_id) const {
    auto it = _variables_state_info.find(variable_id);
    OPENVINO_ASSERT(it != _variables_state_info.end(), "[GPU] ", variable_id, " variable info not found");
    return it->second;
}

const ov::intel_gpu::VariablesMap& network::get_variables() const {
    return _variables_states;
}

const ov::intel_gpu::VariablesInfoMap& network::get_variables_info() const {
    return _variables_state_info;
}

void network::set_variables_state_info(const std::string& variable_id,
                                       const layout& variable_layout,
                                       ov::element::Type user_specified_type,
                                       const primitive* p,
                                       const std::shared_ptr<memory_state::releasable_variable>& releasable_var,
                                       bool transpose_required) {
    auto& info = _variables_state_info.emplace(variable_id, ov::intel_gpu::VariableStateInfo{variable_id, variable_layout, user_specified_type}).first->second;

    [[maybe_unused]] const auto [_, inserted] = info.m_primitives.insert(p);
    if (inserted && releasable_var) {
        info.m_release_variable_inst.emplace_back(releasable_var);
    }
    info.transpose_required = transpose_required;
}

void network::set_reuse_variable_mem(bool reuse) {
    _reuse_variable_mem = reuse;
}


}  // namespace cldnn
