/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

#include <Python.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <thread>

#include "../module.h"
#include "pypto/runtime/instruction.h"
#include "pypto/runtime/machine.h"
#include "pypto/runtime/program.h"

namespace nb = nanobind;
using namespace pypto::runtime;

namespace pypto {
namespace python {

void BindRuntime(nb::module_& m) {
  auto runtime_module = m.def_submodule("runtime", "Runtime machine simulator");

  // Value type binding
  nb::class_<Value>(runtime_module, "Value", "A value that can be held in a register")
      .def(nb::init<>())
      .def(nb::init<int64_t>())
      .def(nb::init<double>())
      .def(nb::init<std::string>());

  // RuntimeProgram
  nb::class_<RuntimeProgram>(runtime_module, "RuntimeProgram", "A program that executes on AICPU")
      .def(nb::init<>())
      .def("add_label", &RuntimeProgram::AddLabel, nb::arg("label"), "Add a label at the current position")
      .def(
          "const",
          [](RuntimeProgram& prog, const std::string& dst, const nb::object& value) {
            // Convert Python object to Value
            Value val;
            if (nb::isinstance<nb::int_>(value)) {
              val = Value(static_cast<int64_t>(nb::cast<int64_t>(value)));
            } else if (nb::isinstance<nb::float_>(value)) {
              val = Value(nb::cast<double>(value));
            } else if (nb::isinstance<nb::str>(value)) {
              val = Value(nb::cast<std::string>(value));
            } else {
              val = Value();
            }
            prog.Const(Register(dst), val);
          },
          nb::arg("dst"), nb::arg("value"), "Add CONST instruction")
      .def(
          "add",
          [](RuntimeProgram& prog, const std::string& dst, const std::string& src1, const std::string& src2) {
            prog.Add(Register(dst), Register(src1), Register(src2));
          },
          nb::arg("dst"), nb::arg("src1"), nb::arg("src2"), "Add ADD instruction")
      .def(
          "sub",
          [](RuntimeProgram& prog, const std::string& dst, const std::string& src1, const std::string& src2) {
            prog.Sub(Register(dst), Register(src1), Register(src2));
          },
          nb::arg("dst"), nb::arg("src1"), nb::arg("src2"), "Add SUB instruction")
      .def(
          "mul",
          [](RuntimeProgram& prog, const std::string& dst, const std::string& src1, const std::string& src2) {
            prog.Mul(Register(dst), Register(src1), Register(src2));
          },
          nb::arg("dst"), nb::arg("src1"), nb::arg("src2"), "Add MUL instruction")
      .def(
          "cmp_eq",
          [](RuntimeProgram& prog, const std::string& dst, const std::string& src1, const std::string& src2) {
            prog.CmpEq(Register(dst), Register(src1), Register(src2));
          },
          nb::arg("dst"), nb::arg("src1"), nb::arg("src2"), "Add CMP_EQ instruction")
      .def("jump", &RuntimeProgram::Jump, nb::arg("label"), "Add JUMP instruction")
      .def(
          "jump_if_zero",
          [](RuntimeProgram& prog, const std::string& cond, const std::string& label) {
            prog.JumpIfZero(Register(cond), label);
          },
          nb::arg("cond"), nb::arg("label"), "Add JUMP_IF_ZERO instruction")
      .def(
          "jump_if_not_zero",
          [](RuntimeProgram& prog, const std::string& cond, const std::string& label) {
            prog.JumpIfNotZero(Register(cond), label);
          },
          nb::arg("cond"), nb::arg("label"), "Add JUMP_IF_NOT_ZERO instruction")
      .def("halt", &RuntimeProgram::Halt, "Add HALT instruction")
      .def(
          "dispatch",
          [](RuntimeProgram& prog, const std::string& handle, const std::string& task_name,
             const std::vector<std::string>& args) {
            std::vector<Register> reg_args;
            for (const auto& arg : args) {
              reg_args.emplace_back(arg);
            }
            prog.Dispatch(Register(handle), task_name, reg_args);
          },
          nb::arg("handle"), nb::arg("task_name"), nb::arg("args"), "Add DISPATCH instruction")
      .def(
          "wait", [](RuntimeProgram& prog, const std::string& handle) { prog.Wait(Register(handle)); },
          nb::arg("handle"), "Add WAIT instruction")
      .def(
          "wait_all",
          [](RuntimeProgram& prog, const std::vector<std::string>& handles) {
            std::vector<Register> reg_handles;
            for (const auto& handle : handles) {
              reg_handles.emplace_back(handle);
            }
            prog.WaitAll(reg_handles);
          },
          nb::arg("handles"), "Add WAIT_ALL instruction")
      .def(
          "store_mem",
          [](RuntimeProgram& prog, int64_t addr, const std::string& reg) {
            prog.StoreMem(addr, Register(reg));
          },
          nb::arg("addr"), nb::arg("reg"), "Add STORE_MEM instruction")
      .def(
          "load_mem",
          [](RuntimeProgram& prog, const std::string& reg, int64_t addr) {
            prog.LoadMem(Register(reg), addr);
          },
          nb::arg("reg"), nb::arg("addr"), "Add LOAD_MEM instruction")
      .def("nop", &RuntimeProgram::Nop, "Add NOP instruction")
      .def("to_python_syntax", &RuntimeProgram::ToPythonSyntax, "Convert program to Python-like syntax")
      .def("get_instruction_count", &RuntimeProgram::GetInstructionCount, "Get total number of instructions")
      .def_static(
          "reg", [](int n) { return RuntimeProgram::Reg(n).ToString(); }, nb::arg("n"),
          "Generate register name from integer (e.g., reg(0) -> 'r0')")
      .def("__str__", &RuntimeProgram::ToPythonSyntax)
      .def("__repr__", [](const RuntimeProgram& prog) {
        return "<RuntimeProgram with " + std::to_string(prog.GetInstructionCount()) + " instructions>";
      });

  // SharedMemory
  nb::class_<SharedMemory>(runtime_module, "SharedMemory", "Shared memory for AICPU-AICORE communication")
      .def("write", &SharedMemory::Write, nb::arg("addr"), nb::arg("value"), "Write value to memory")
      .def("read", &SharedMemory::Read, nb::arg("addr"), "Read value from memory")
      .def("contains", &SharedMemory::Contains, nb::arg("addr"), "Check if address exists")
      .def("clear", &SharedMemory::Clear, "Clear all memory");

  // RuntimeMachine
  nb::class_<RuntimeMachine>(runtime_module, "RuntimeMachine",
                             "Simulated runtime machine (AICPU host with AICORE workers)")
      .def(nb::init<int>(), nb::arg("num_aicore"),
           "Create a runtime machine with specified number of AICORE workers")
      .def(
          "register_task",
          [](RuntimeMachine& machine, const std::string& name, const nb::object& callable) {
            // Convert to raw PyObject* to avoid nanobind reference counting issues across threads
            // We manually manage the refcount with GIL protection
            PyObject* py_callable = callable.ptr();
            Py_INCREF(py_callable);  // Increment refcount to keep it alive

            // Wrap Python callable in a C++ function
            machine.RegisterTask(name, [py_callable](const std::vector<Value>& args) -> Value {
              // Acquire GIL before calling Python from C++ thread
              nb::gil_scoped_acquire gil;

              try {
                // Wrap raw PyObject* back to nb::object
                nb::object callable = nb::borrow(py_callable);

                // Convert C++ args to Python objects
                nb::list py_args;
                for (const auto& arg : args) {
                  if (std::holds_alternative<int64_t>(arg)) {
                    py_args.append(std::get<int64_t>(arg));
                  } else if (std::holds_alternative<double>(arg)) {
                    py_args.append(std::get<double>(arg));
                  } else if (std::holds_alternative<std::string>(arg)) {
                    py_args.append(std::get<std::string>(arg));
                  } else {
                    py_args.append(nb::none());
                  }
                }

                // Call Python function
                nb::object result = callable(*py_args);

                // Convert result back to Value
                if (nb::isinstance<nb::int_>(result)) {
                  return Value(static_cast<int64_t>(nb::cast<int64_t>(result)));
                } else if (nb::isinstance<nb::float_>(result)) {
                  return Value(nb::cast<double>(result));
                } else if (nb::isinstance<nb::str>(result)) {
                  return Value(nb::cast<std::string>(result));
                } else {
                  return Value();
                }
              } catch (const std::exception& e) {
                std::cerr << "Python callable execution failed: " << e.what() << '\n';
                return Value();
              }
            });
          },
          nb::arg("name"), nb::arg("callable"), "Register a task implementation")
      .def(
          "load_and_run_program",
          [](RuntimeMachine& machine, RuntimeProgram& program) {
            // Make a copy of the program and wrap in shared_ptr
            auto prog_copy = std::make_shared<RuntimeProgram>(program);

            // Run in a separate thread to avoid GIL deadlock
            // This allows AICORE worker threads to acquire GIL for Python callables
            std::exception_ptr exception_ptr = nullptr;
            std::atomic<bool> finished{false};

            std::thread execution_thread([&]() {
              try {
                machine.LoadAndRunProgram(prog_copy);
              } catch (...) {
                exception_ptr = std::current_exception();
              }
              finished.store(true);
            });

            // Wait for completion while periodically releasing GIL
            while (!finished.load()) {
              nb::gil_scoped_release release;
              std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            execution_thread.join();

            // Re-throw exception if one occurred
            if (exception_ptr) {
              std::rethrow_exception(exception_ptr);
            }
          },
          nb::arg("program"),
          "Load and execute a program synchronously on AICPU host (blocks until completion)")
      .def(
          "get_memory", [](RuntimeMachine& machine) -> SharedMemory& { return *machine.GetMemory(); },
          nb::rv_policy::reference, "Get shared memory")
      .def("get_num_aicore", &RuntimeMachine::GetNumAICORE, "Get number of AICOREs")
      .def("__repr__", [](const RuntimeMachine& machine) {
        return "<RuntimeMachine (AICPU host) with " + std::to_string(machine.GetNumAICORE()) + " AICOREs>";
      });
}

}  // namespace python
}  // namespace pypto
