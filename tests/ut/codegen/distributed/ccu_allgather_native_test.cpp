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

// Execute the emitted native definition with CPU primitive stubs. This checks
// address/token wiring, completion ordering and BF16 bits, not HCOMM synthesis.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include "platform_comm/ccu/ccu_native_abi.h"

namespace simpler::ccu {
inline constexpr uint8_t sdk_build_id[32] = {0x12};
inline constexpr uint8_t pto_isa_build_id[32] = {0x34};
}  // namespace simpler::ccu

enum CcuResult { CCU_SUCCESS, CCU_E_INTERNAL };
namespace model {
std::array<uint64_t, 9 + 3 * TEST_RANKS> words;
std::array<uint32_t, TEST_RANKS> physical_ranks;
std::array<uint32_t, TEST_RANKS> peer_slots;
std::array<bool, TEST_RANKS> written;
std::array<uint16_t, 16> transfer_bits;
std::array<uint64_t, TEST_RANKS> copied_bytes, output_bases;
uint64_t input_base, block_bytes;
uint32_t event_offset;
int main_groups, remainder_groups;
uint64_t peers, records, waits;
uint32_t source_rank, next_slot;
int phase, done_count, trigger_budget, trigger_waits;
struct StreamClosed {};
bool stream_closed;
bool fail_local_copy, dry_run;

uint64_t ChannelBit(uint64_t channel) {
  assert(channel >= 1000 && channel < 1064);
  const auto bit = uint64_t{1} << (channel - 1000);
  assert(peers & bit);
  return bit;
}
}  // namespace model

namespace AscendC::ccu {
struct Variable {
  uint64_t value = 0;
  void operator=(uint64_t v) { value = v; }
  Variable operator+(const Variable& rhs) const { return {value + rhs.value}; }
  bool operator!=(uint64_t rhs) const { return value != rhs; }
  // CANN comparison operators construct expressions from mutable Variables.
  bool operator==(uint64_t rhs) { return value == rhs; }
};
struct Address {
  std::shared_ptr<uint64_t> cell = std::make_shared<uint64_t>(0);
  Address() = default;
  Address(const Address&) = default;
  Address& operator=(const Address& rhs) {
    *cell = *rhs.cell;
    return *this;
  }
  void operator=(const Variable& rhs) { *cell = rhs.value; }
  void operator+=(const Variable& rhs) { *cell += rhs.value; }
  Address operator+(const Variable& rhs) const {
    Address result;
    *result.cell = *cell + rhs.value;
    return result;
  }
};
struct LocalAddr {
  Address addr;
  Variable token;
};
struct RemoteAddr {
  Address addr;
  Variable token;
};
struct Event {
  uint64_t handle;
  explicit Event(uint64_t h = 100) : handle(h) {}
};
template <typename T>
struct Array {
  std::vector<T> items;
  explicit Array(uint32_t count) : items(count) {
    assert(count == 16);
    for (uint32_t i = 0; i < count; ++i) items[i].handle = 100 + i;
  }
  T operator[](uint32_t index) { return items[index]; }
};
struct Func;
Func* capturing = nullptr;
struct Func {
  std::vector<std::function<void()>> ops;
  std::vector<std::shared_ptr<uint64_t>> addresses;
  explicit Func(const std::function<void()>& body) {
    assert(capturing == nullptr);
    capturing = this;
    body();
    capturing = nullptr;
  }
  void AddressUsed(const Address& address) {
    if (std::find(addresses.begin(), addresses.end(), address.cell) == addresses.end())
      addresses.push_back(address.cell);
  }
};
struct Loop {
  uint64_t param;
  Func body;
  std::shared_ptr<bool> attached = std::make_shared<bool>(false);
  Loop(const Variable& value, const Func& func) : param(value.value), body(func) {}
};
struct LoopGroup {
  LoopGroup(const Variable& parallel, const Variable& offsets, uint32_t max_loops, std::vector<Loop>& loops) {
    const uint32_t repeat = (parallel.value >> 55) & 127;
    const uint32_t index = (parallel.value >> 48) & 127;
    const uint32_t total = (parallel.value >> 41) & 127;
    assert(total >= 1 && total <= max_loops && loops.size() == max_loops && index < total);
    assert(((offsets.value >> 10) & 2047) == 0);  // no MS banks
    assert((offsets.value & 1023) == 1);          // consecutive per-lane CKEs
    const uint64_t lane_stride = offsets.value >> 21;
    assert(lane_stride == 32768);
    if (max_loops == 1) {
      assert(repeat == 15 && index == 0);
      ++model::main_groups;
    } else {
      assert(max_loops == 2);
      ++model::remainder_groups;
    }
    for (auto& loop : loops) {
      assert(!*loop.attached);  // one Loop object must not join multiple groups
      *loop.attached = true;
    }
    // Interpret public LoopGroup fields and advance the addresses captured by
    // each Func. This models schedule coverage, not HCOMM synthesis or timing.
    for (uint32_t i = 0; i < total; ++i) {
      auto& loop = loops[i];
      const uint64_t iterations = loop.param & 8191;
      const uint64_t stride = loop.param >> 13;
      const uint32_t lanes = i == index ? repeat + 1 : 1;
      assert(lanes + i <= 16);
      std::vector<uint64_t> bases;
      for (const auto& address : loop.body.addresses) bases.push_back(*address);
      for (uint64_t iteration = 0; iteration < iterations; ++iteration) {
        for (uint32_t lane = 0; lane < lanes; ++lane) {
          for (size_t a = 0; a < bases.size(); ++a)
            *loop.body.addresses[a] = bases[a] + iteration * stride + lane * lane_stride;
          model::event_offset = lane;
          for (const auto& op : loop.body.ops) op();
        }
      }
      // Leave walked addresses advanced, as hardware can. Remainder/request
      // rebinding must use stable bases rather than these mutated handles.
      for (size_t a = 0; a < bases.size(); ++a)
        *loop.body.addresses[a] = bases[a] + iterations * stride + lanes * lane_stride;
    }
    model::event_offset = 0;
  }
};
inline CcuResult LoadArg(Variable& dst, uint32_t index) {
  assert(index < SIMPLER_CCU_TASK_ARGUMENT_COUNT);
  dst = index == 0 ? reinterpret_cast<uint64_t>(model::words.data()) : 999;
  return CCU_SUCCESS;
}
inline CcuResult Load(const Variable& address, Variable& dst) {
  dst = *reinterpret_cast<const uint64_t*>(address.value);
  return CCU_SUCCESS;
}
inline CcuResult EventWait(const Event& event, uint16_t mask) {
  if (capturing) {
    capturing->ops.push_back([event, mask]() { EventWait(event, mask); });
    return CCU_SUCCESS;
  }
  if (event.handle == 11) {
    assert(mask == 1 && model::phase == 0);
    ++model::trigger_waits;
    if (model::done_count == model::trigger_budget) {
      // Model Host shutdown while the resident mission is parked at trigger.
      // This is only an interpreter escape: real stream closure happens on the
      // device, not by throwing through the host instruction-builder callback.
      model::stream_closed = true;
      throw model::StreamClosed{};
    }
    model::phase = 1;
  } else {
    const auto lane = event.handle - 100 + model::event_offset;
    assert(lane < 16 && model::phase == 3);
    assert(model::next_slot % 16 == 0 || model::next_slot == TEST_RANKS);
    const auto batch = (model::next_slot - 1) % 16 + 1;
    assert(mask == (1u << batch) - 1 && model::transfer_bits[lane] == mask);
    model::transfer_bits[lane] = 0;
    if (model::next_slot == TEST_RANKS) model::next_slot = 0;
  }
  return CCU_SUCCESS;
}
inline CcuResult EventRecord(const Event& event, uint16_t mask) {
  assert(event.handle == 12 && mask == 1 && model::phase == 5);
  assert(model::waits == model::peers);
  for (const auto bytes : model::copied_bytes) assert(bytes == model::block_bytes);
  model::copied_bytes.fill(0);
  ++model::done_count;
  model::phase = 0;
  model::next_slot = 0;
  model::records = model::waits = 0;
  return CCU_SUCCESS;
}
inline CcuResult NotifyRecord(uint64_t channel, uint32_t index, uint16_t mask) {
  assert(index == 0);
  if (mask == 1) {
    assert(model::phase == 1);
  } else {
    assert(mask == 2 && model::next_slot == 0);
    for (const auto bits : model::transfer_bits) assert(bits == 0);
    for (const auto bytes : model::copied_bytes) assert(bytes == model::block_bytes);
    if (model::phase == 3) model::records = model::waits = 0;
    assert(model::phase == 3 || model::phase == 4);
    model::phase = 4;
  }
  const auto bit = model::ChannelBit(channel);
  assert(!(model::records & bit));
  model::records |= bit;
  return CCU_SUCCESS;
}
inline CcuResult NotifyWait(uint64_t channel, uint32_t index, uint16_t mask) {
  assert(index == 0 && model::records == model::peers);
  assert((mask == 1 && (model::phase == 1 || model::phase == 2)) ||
         (mask == 2 && (model::phase == 4 || model::phase == 5)));
  const auto bit = model::ChannelBit(channel);
  assert(!(model::waits & bit));
  model::waits |= bit;
  model::phase = mask == 1 ? 2 : 5;
  return CCU_SUCCESS;
}
inline void Copy(const Address& dst, const LocalAddr& src, const Variable& bytes, const Event& event,
                 uint16_t mask) {
  assert(model::phase == 2 || model::phase == 3);
  assert(model::waits == model::peers && src.token.value == 101);
  const auto lane = event.handle - 100 + model::event_offset;
  assert(lane < 16 && mask == (1u << (model::next_slot % 16)));
  assert(!(model::transfer_bits[lane] & mask));
  const auto slot = model::peer_slots[model::next_slot];
  const auto offset = *src.addr.cell - model::input_base;
  assert(offset == model::copied_bytes[slot]);
  assert(offset + bytes.value <= model::block_bytes);
  assert(*dst.cell == model::output_bases[slot] + model::source_rank * model::block_bytes + offset);
  if (!model::dry_run)
    std::memcpy(reinterpret_cast<void*>(*dst.cell), reinterpret_cast<const void*>(*src.addr.cell),
                bytes.value);
  model::copied_bytes[slot] += bytes.value;
  model::written[slot] = true;
  ++model::next_slot;
  model::transfer_bits[lane] |= mask;
  model::phase = 3;
}
inline CcuResult Write(uint64_t channel, const RemoteAddr& dst, const LocalAddr& src, const Variable& bytes,
                       const Event& event, uint16_t mask) {
  if (capturing) {
    capturing->AddressUsed(dst.addr);
    capturing->AddressUsed(src.addr);
    capturing->ops.push_back(
        [channel, dst, src, bytes, event, mask]() { Write(channel, dst, src, bytes, event, mask); });
    return CCU_SUCCESS;
  }
  const auto slot = model::peer_slots[model::next_slot];
  assert(slot != model::source_rank);
  assert(channel == 1000 + model::physical_ranks[slot]);
  assert(dst.token.value == 200 + slot);
  Copy(dst.addr, src, bytes, event, mask);
  return CCU_SUCCESS;
}
inline CcuResult LocalCopy(const LocalAddr& dst, const LocalAddr& src, const Variable& bytes,
                           const Event& event, uint16_t mask) {
  if (capturing) {
    capturing->AddressUsed(dst.addr);
    capturing->AddressUsed(src.addr);
    capturing->ops.push_back([dst, src, bytes, event, mask]() { LocalCopy(dst, src, bytes, event, mask); });
    return CCU_SUCCESS;
  }
  assert(model::peer_slots[model::next_slot] == model::source_rank &&
         dst.token.value == 200 + model::source_rank);
  if (model::fail_local_copy) throw CCU_E_INTERNAL;
  Copy(dst.addr, src, bytes, event, mask);
  return CCU_SUCCESS;
}
}  // namespace AscendC::ccu
#define CCU_WHILE(condition) while (condition)
#define CCU_IF(condition) if ((assert(AscendC::ccu::capturing == nullptr), (condition)))
#include "generated_module.inc"

void RunCase(uint32_t rank, uint64_t count, bool failure, bool dry_run, int triggers = 3,
             bool permuted = false) {
  std::vector<uint16_t> input((dry_run ? 0 : count) + 2);
  // Large cases store only the touched row and its canaries. Copy checks
  // virtual target offsets before accessing memory, so this still catches
  // wrong-row writes without allocating a quadratic rank-by-rank matrix.
  const uint64_t stored_rows = count > 64 ? 1 : TEST_RANKS;
  std::vector<std::vector<uint16_t>> outputs(
      TEST_RANKS, std::vector<uint16_t>((dry_run ? 0 : stored_rows * count) + 6, 0xdead));
  for (uint64_t i = 0; !dry_run && i < count; ++i) input[2 + i] = static_cast<uint16_t>(0x7f80 + i);
  model::words[0] = reinterpret_cast<uint64_t>(input.data() + 2);
  model::words[1] = 101;
  model::words[2] = count * 2;
  model::words[3] = rank * count * 2;
  model::words[4] = count * 2 / pypto_ccu::kSegmentBytes;
  const auto go = pypto_ccu::CalcGoSize(count * 2 % pypto_ccu::kSegmentBytes);
  model::words[5] = go.addr_offset;
  model::words[6] = go.iterations;
  model::words[7] = go.parallel;
  model::words[8] = go.residual;
  model::phase = model::done_count = model::main_groups = model::remainder_groups = 0;
  model::trigger_budget = triggers;
  model::trigger_waits = 0;
  model::stream_closed = false;
  model::transfer_bits.fill(0);
  model::copied_bytes.fill(0);
  model::input_base = model::words[0];
  model::block_bytes = count * 2;
  model::next_slot = 0;
  model::source_rank = rank;
  model::peers = model::records = model::waits = 0;
  model::written.fill(false);
  model::fail_local_copy = failure;
  model::dry_run = dry_run;
  SimplerCcuRegistrationBinding binding{};
  binding.rank_count = TEST_RANKS;
  binding.ccu_id = 7;  // Mission routing must not determine the peer notify index.
  binding.trigger_event_handle = 11;
  binding.done_event_handle = 12;
  for (uint32_t slot = 0; slot < TEST_RANKS; ++slot) {
    // Permuted, noncontiguous subsets catch domain/physical rank confusion.
    const uint32_t physical = (63 - 5 * slot) % 64;
    const uint32_t peer = permuted ? TEST_RANKS - 1 - slot : slot;
    model::physical_ranks[slot] = physical;
    model::peer_slots[peer] = slot;
    model::words[9 + 3 * slot] =
        reinterpret_cast<uint64_t>(outputs[slot].data() + 3) - (count > 64 ? rank * count * 2 : 0);
    model::words[10 + 3 * slot] = 200 + slot;
    model::words[11 + 3 * slot] = peer;
    model::output_bases[slot] = model::words[9 + 3 * slot];
    binding.channels[peer] = 1000 + physical;
    if (slot == rank)
      binding.rank_id = peer;
    else
      model::peers |= uint64_t{1} << physical;
  }
  const auto result = pypto_ccu_allgather_test(&binding);
  // The native callback catches the interpreter's stream-closure exception.
  // CCU_E_INTERNAL here is not a real stream-abort return code.
  assert(result == CCU_E_INTERNAL);
  assert(model::stream_closed == !failure);
  assert(model::done_count == (failure ? 0 : triggers));
  assert(model::trigger_waits == (failure ? 1 : triggers + 1));
  if (!failure) {
    assert(model::main_groups == triggers * (model::words[4] + (go.iterations ? 1 : 0)));
    assert(model::remainder_groups == triggers * (model::words[4] + (go.parallel ? 1 : 0)));
  }
  // Every destination gets exactly this source rank's row. Nonzero base
  // offsets and sentinels catch double slot offsets and neighbor corruption.
  for (uint32_t slot = 0; slot < TEST_RANKS; ++slot) {
    const auto peer = model::words[11 + 3 * slot];
    assert(model::written[slot] == ((!failure && triggers > 0) || (failure && peer < binding.rank_id)));
    for (uint64_t i = 0; !dry_run && i < outputs[slot].size(); ++i) {
      const uint64_t row_start = 3 + (count > 64 ? 0 : rank * count);
      const bool copied = model::written[slot] && i >= row_start && i < row_start + count;
      const auto expected = copied ? input[2 + i - row_start] : uint16_t{0xdead};
      assert(outputs[slot][i] == expected);
    }
  }
}

int main() {
  const auto* manifest = simpler_ccu_module_manifest();
  const auto* base = reinterpret_cast<const uint8_t*>(manifest);
  const auto* kernel =
      reinterpret_cast<const SimplerCcuKernelRecord*>(base + manifest->kernel_records_offset);
  assert(manifest->struct_size == sizeof(SimplerCcuModuleManifest) && manifest->kernel_count == 1);
  assert(manifest->kernel_record_size == sizeof(SimplerCcuKernelRecord));
  assert(manifest->abi_version == SIMPLER_CCU_ABI_VERSION && manifest->reserved == 0);
  assert(kernel->required_rank_count == TEST_RANKS && kernel->reserved == 0);
  assert(manifest->sdk_build_id[0] == 0x12 && manifest->pto_isa_build_id[0] == 0x34);
  assert(kernel->parameter_size == (9 + 3 * TEST_RANKS) * 8 && kernel->parameter_alignment == 64);
  assert(kernel->registration_argument_count == 1 && kernel->task_argument_count == 2);
  assert(kernel->capability_flags == SIMPLER_CCU_KERNEL_CAP_THREE_EVENTS);
  assert(std::memcmp(base + kernel->entry_symbol_offset, "pypto_ccu_allgather_test",
                     kernel->entry_symbol_size) == 0);
  for (uint32_t rank = 0; rank < TEST_RANKS; ++rank) {
    for (bool failure : {false, true}) RunCase(rank, 7, failure, false, 3, true);
    for (uint64_t count : {7U, 64U, 16384U, 16391U, 32767U, 262144U, 262151U, 278528U, 278535U, 524288U}) {
      // Large rows cover main/remainder combinations at both domain endpoints;
      // small rows still exercise every possible source rank and failure slot.
      if (count > 64 && rank != 0 && rank + 1 != TEST_RANKS) continue;
      for (bool failure : {false, true}) {
        if (failure && count > 64) continue;
        RunCase(rank, count, failure, false);
      }
    }
  }
  // No trigger, one trigger and more triggers than any static call-site count
  // all remain parked until the external shutdown, with no extra done/exit task.
  for (int triggers : {0, 1, 5}) RunCase(TEST_RANKS - 1, 7, false, false, triggers);
  if (TEST_RANKS == 2) {
    for (uint64_t bytes : {pypto_ccu::kSegmentBytes - 2, pypto_ccu::kSegmentBytes,
                           pypto_ccu::kSegmentBytes + 32770, 2 * pypto_ccu::kSegmentBytes + 14}) {
      RunCase(1, bytes / 2, false, true);
    }
  }
}
