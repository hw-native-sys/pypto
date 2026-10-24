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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_STORAGE_ACCESS_INDEX_H_
#define PYPTO_IR_TRANSFORMS_UTILS_STORAGE_ACCESS_INDEX_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/scalar_expr.h"

namespace pypto::ir {

namespace storage_access_detail {

// Offsets are signed 64-bit and extents unsigned 64-bit. Their sum needs
// a wider intermediate even when the allocation itself is well formed.
using Offset = __int128;
struct ByteRange {
  Offset begin;
  Offset end;
};

inline std::optional<uint64_t> StaticSlot(const MemRefPtr& window) {
  if (window->slot_count_ == 0) return std::nullopt;
  if (!window->slot_index_ || !*window->slot_index_) {
    return window->slot_count_ == 1 ? std::make_optional(uint64_t{0}) : std::nullopt;
  }
  const auto index = As<ConstInt>(*window->slot_index_);
  if (!index || index->value_ < 0 || static_cast<uint64_t>(index->value_) >= window->slot_count_) {
    return std::nullopt;
  }
  return static_cast<uint64_t>(index->value_);
}

inline std::optional<ByteRange> StaticBytes(const MemRefPtr& window) {
  if (!window) return std::nullopt;
  const auto offset = As<ConstInt>(window->byte_offset_);
  if (!offset) return std::nullopt;
  const Offset begin = offset->value_;
  return ByteRange{begin, begin + static_cast<Offset>(window->size_)};
}

}  // namespace storage_access_detail

/// Immutable storage-access event index in a frozen control-flow snapshot.
///
/// A query keeps only events after its primary boundary, then checks the queried
/// byte window and slot. The minimum primary and maximum reverse positions are
/// conservative envelopes: they may belong to different events. Consequently a
/// positive answer only indicates possible overlap; it does not prove one event
/// satisfies every bound. A negative answer proves no indexed event qualifies.
///
/// Build costs O(N log N) time and storage; MayOverlapAfter costs O(log N). Each
/// event occupies at most two slot buckets. Callers select the reads or writes
/// to index and must supply their actual windows before Build. Rebuild
/// after snapshot changes; this analysis does not observe later IR mutations.
/// An omitted optional access must precede the query boundary or be mutually
/// exclusive with it; later source mutations need their separate mutable-storage
/// constraints. This index alone does not justify changing its snapshot.
class StorageAccessIndex {
 public:
  struct Event {
    MemRefPtr window;
    size_t primary;
    size_t reverse;
  };

  void Build(const std::vector<Event>& events) {
    bases_.clear();
    buckets_.clear();
    versions_.clear();
    nodes_.assign(1, Node{});  // Index zero is the empty persistent subtree.
    total_cells_ = 0;
    unknown_bucket_ = kNoBucket;

    // Slot identities are useful only when every event agrees on the slot
    // count. Mismatched counts use the all-slots bucket rather than proving
    // separation from contradictory metadata.
    for (const auto& event : events) {
      if (!event.window || !event.window->base_ || event.window->size_ == 0) continue;
      const auto& window = event.window;
      auto [found, inserted] = bases_.try_emplace(window->base_.get());
      if (inserted) {
        found->second.owner = window->base_;
        found->second.slot_count = window->slot_count_;
        found->second.consistent_slots = window->slot_count_ != 0;
      } else if (found->second.slot_count != window->slot_count_) {
        found->second.consistent_slots = false;
      }
    }

    std::vector<PendingEvent> pending;
    for (const auto& event : events) {
      if (event.window && event.window->size_ == 0) continue;
      if (!event.window || !event.window->base_) {
        // Missing allocation identity cannot be attributed to one base. Keep
        // its event as a global opaque cell instead of silently dropping it.
        AddEvent(EnsureBucket(unknown_bucket_), event, &pending);
        continue;
      }
      auto& base = bases_.at(event.window->base_.get());
      AddEvent(EnsureBucket(base.all_slots), event, &pending);
      if (!base.consistent_slots) continue;
      if (const auto slot = StaticSlot(event.window)) {
        auto [found, inserted] = base.slots.try_emplace(*slot, kNoBucket);
        (void)inserted;
        AddEvent(EnsureBucket(found->second), event, &pending);
      } else {
        AddEvent(EnsureBucket(base.wildcard_slot), event, &pending);
      }
    }

    // Every bucket has its own compressed byte cells. Flattening those cells
    // gives one persistent tree without joining unrelated allocation windows.
    for (auto& bucket : buckets_) {
      auto& coordinates = bucket.coordinates;
      std::sort(coordinates.begin(), coordinates.end());
      coordinates.erase(std::unique(coordinates.begin(), coordinates.end()), coordinates.end());
      bucket.first_cell = total_cells_;
      bucket.static_cells = coordinates.empty() ? 0 : coordinates.size() - 1;
      total_cells_ += bucket.static_cells;
      if (bucket.has_opaque) bucket.opaque_cell = total_cells_++;
    }
    if (pending.empty()) return;

    std::sort(pending.begin(), pending.end(),
              [](const PendingEvent& lhs, const PendingEvent& rhs) { return lhs.primary > rhs.primary; });
    size_t root = 0;
    for (size_t begin = 0; begin < pending.size();) {
      const size_t primary = pending[begin].primary;
      size_t end = begin;
      for (; end < pending.size() && pending[end].primary == primary; ++end) {
        const auto& event = pending[end];
        const auto& bucket = buckets_[event.bucket];
        CellRange range;
        if (event.bytes) {
          const auto first =
              std::lower_bound(bucket.coordinates.begin(), bucket.coordinates.end(), event.bytes->begin);
          const auto last =
              std::lower_bound(bucket.coordinates.begin(), bucket.coordinates.end(), event.bytes->end);
          range = {bucket.first_cell + static_cast<size_t>(first - bucket.coordinates.begin()),
                   bucket.first_cell + static_cast<size_t>(last - bucket.coordinates.begin())};
        } else {
          range = {bucket.opaque_cell, bucket.opaque_cell + 1};
        }
        root = Update(root, 0, total_cells_, range, Summary{true, event.primary, event.reverse});
      }
      // Publish after all equal-primary events, so a strict-after query can
      // never include only a subset of the events at one statement boundary.
      versions_.push_back({primary, root});
      begin = end;
    }
  }

  /// May an access overlap after both boundaries and before the upper bound?
  /// The upper primary bound is inclusive; max(size_t) means the whole suffix.
  /// A null allocation/window conservatively considers every indexed base.
  [[nodiscard]] bool MayOverlapAfter(const MemRefPtr& window, size_t after_primary, size_t after_reverse,
                                     size_t last_use_primary = std::numeric_limits<size_t>::max()) const {
    if (last_use_primary <= after_primary || versions_.empty() || (window && window->size_ == 0)) {
      return false;
    }
    const auto boundary =
        std::lower_bound(versions_.begin(), versions_.end(), after_primary,
                         [](const Version& version, size_t cutoff) { return version.primary > cutoff; });
    if (boundary == versions_.begin()) return false;
    const size_t root = std::prev(boundary)->root;
    Summary summary;
    if (!window || !window->base_) {
      summary = Query(root, 0, total_cells_, {0, total_cells_});
    } else {
      summary = QueryBucket(root, unknown_bucket_, std::nullopt);
      if (const auto found = bases_.find(window->base_.get()); found != bases_.end()) {
        const auto& base = found->second;
        const auto bytes = StaticBytes(window);
        const auto slot = StaticSlot(window);
        if (base.consistent_slots && window->slot_count_ == base.slot_count && slot) {
          if (const auto entry = base.slots.find(*slot); entry != base.slots.end()) {
            summary.Include(QueryBucket(root, entry->second, bytes));
          }
          summary.Include(QueryBucket(root, base.wildcard_slot, bytes));
        } else {
          summary.Include(QueryBucket(root, base.all_slots, bytes));
        }
      }
    }
    return summary.present && summary.min_primary <= last_use_primary && summary.max_reverse > after_reverse;
  }

 private:
  static constexpr size_t kNoBucket = std::numeric_limits<size_t>::max();
  using Offset = storage_access_detail::Offset;
  using ByteRange = storage_access_detail::ByteRange;
  struct CellRange {
    size_t begin;
    size_t end;
  };
  struct Summary {
    bool present = false;
    size_t min_primary = std::numeric_limits<size_t>::max();
    size_t max_reverse = 0;

    void Include(const Summary& other) {
      if (!other.present) return;
      present = true;
      min_primary = std::min(min_primary, other.min_primary);
      max_reverse = std::max(max_reverse, other.max_reverse);
    }
  };
  struct Node {
    size_t left = 0;
    size_t right = 0;
    Summary cover;  // Events covering the complete node range.
    Summary all;    // Cover plus events in either descendant.
  };
  struct Bucket {
    std::vector<Offset> coordinates;
    size_t first_cell = 0;
    size_t static_cells = 0;
    size_t opaque_cell = kNoBucket;
    bool has_opaque = false;
  };
  struct Base {
    VarPtr owner;  // Keep the identity token alive for the snapshot's lifetime.
    uint64_t slot_count = 0;
    bool consistent_slots = false;
    size_t all_slots = kNoBucket;
    size_t wildcard_slot = kNoBucket;
    std::map<uint64_t, size_t> slots;
  };
  struct PendingEvent {
    size_t bucket;
    std::optional<ByteRange> bytes;
    size_t primary;
    size_t reverse;
  };
  struct Version {
    size_t primary;
    size_t root;
  };

  static std::optional<uint64_t> StaticSlot(const MemRefPtr& window) {
    return storage_access_detail::StaticSlot(window);
  }

  static std::optional<ByteRange> StaticBytes(const MemRefPtr& window) {
    return storage_access_detail::StaticBytes(window);
  }

  size_t EnsureBucket(size_t& bucket) {
    if (bucket == kNoBucket) {
      bucket = buckets_.size();
      buckets_.emplace_back();
    }
    return bucket;
  }

  void AddEvent(size_t bucket, const Event& event, std::vector<PendingEvent>* pending) {
    const auto bytes = event.window && event.window->base_ ? StaticBytes(event.window) : std::nullopt;
    if (bytes) {
      buckets_[bucket].coordinates.push_back(bytes->begin);
      buckets_[bucket].coordinates.push_back(bytes->end);
    } else {
      buckets_[bucket].has_opaque = true;
    }
    pending->push_back({bucket, bytes, event.primary, event.reverse});
  }

  size_t Update(size_t previous, size_t begin, size_t end, CellRange range, const Summary& event) {
    const size_t current = nodes_.size();
    nodes_.push_back(nodes_[previous]);
    if (range.begin <= begin && end <= range.end) {
      nodes_[current].cover.Include(event);
      nodes_[current].all.Include(event);
      return current;
    }
    const size_t middle = begin + (end - begin) / 2;
    // Recursive insertion may reallocate nodes_; retain indices, not references.
    if (range.begin < middle) {
      const size_t child = Update(nodes_[current].left, begin, middle, range, event);
      nodes_[current].left = child;
    }
    if (middle < range.end) {
      const size_t child = Update(nodes_[current].right, middle, end, range, event);
      nodes_[current].right = child;
    }
    Summary summary = nodes_[current].cover;
    summary.Include(nodes_[nodes_[current].left].all);
    summary.Include(nodes_[nodes_[current].right].all);
    nodes_[current].all = summary;
    return current;
  }

  [[nodiscard]] Summary Query(size_t node, size_t begin, size_t end, CellRange range) const {
    if (node == 0 || range.begin >= range.end || range.end <= begin || end <= range.begin) return {};
    if (range.begin <= begin && end <= range.end) return nodes_[node].all;
    Summary summary = nodes_[node].cover;
    const size_t middle = begin + (end - begin) / 2;
    if (range.begin < middle) summary.Include(Query(nodes_[node].left, begin, middle, range));
    if (middle < range.end) summary.Include(Query(nodes_[node].right, middle, end, range));
    return summary;
  }

  [[nodiscard]] Summary QueryBucket(size_t root, size_t index, const std::optional<ByteRange>& bytes) const {
    if (index == kNoBucket) return {};
    const auto& bucket = buckets_[index];
    Summary summary;
    if (bucket.opaque_cell != kNoBucket) {
      summary = Query(root, 0, total_cells_, {bucket.opaque_cell, bucket.opaque_cell + 1});
    }
    if (bucket.static_cells == 0) return summary;
    if (!bytes) {
      summary.Include(
          Query(root, 0, total_cells_, {bucket.first_cell, bucket.first_cell + bucket.static_cells}));
      return summary;
    }
    const auto& coordinates = bucket.coordinates;
    if (bytes->end <= coordinates.front() || coordinates.back() <= bytes->begin) return summary;
    const auto first = std::upper_bound(coordinates.begin(), coordinates.end(), bytes->begin);
    const auto last = std::lower_bound(coordinates.begin(), coordinates.end(), bytes->end);
    const size_t left =
        first == coordinates.begin() ? 0 : static_cast<size_t>(first - coordinates.begin() - 1);
    const size_t right = std::min(bucket.static_cells, static_cast<size_t>(last - coordinates.begin()));
    summary.Include(Query(root, 0, total_cells_, {bucket.first_cell + left, bucket.first_cell + right}));
    return summary;
  }

  std::map<const Var*, Base> bases_;
  std::vector<Bucket> buckets_;
  std::vector<Node> nodes_{Node{}};
  std::vector<Version> versions_;  // Descending primary; each root includes its suffix.
  size_t total_cells_ = 0;
  size_t unknown_bucket_ = kNoBucket;
};

/// Byte/slot routing shared by incremental and frozen observation indexes.
/// Each interval visits a fixed number of nodes in the 65-bit offset domain.
/// Payloads on ancestor covers and fully covered subtrees preserve overlap
/// without joining disjoint windows through a whole-allocation handle.
template <typename Payload>
class StorageWindowIndex {
 public:
  void Clear() {
    bases_.clear();
    opaque_ = {};
    all_ = {};
    nodes_.assign(1, Node{});
  }

  template <typename Update>
  void UpdateWindow(const MemRefPtr& window, const Update& update) {
    if (window && window->size_ == 0) return;
    update(all_);
    if (!window || !window->base_) {
      update(opaque_);
      return;
    }
    auto [entry, inserted] = bases_.try_emplace(window->base_.get());
    auto& base = entry->second;
    if (inserted) {
      base.owner = window->base_;
      base.slot_count = window->slot_count_;
      base.consistent_slots = window->slot_count_ != 0;
    } else if (base.slot_count != window->slot_count_) {
      base.consistent_slots = false;
    }
    UpdateBucket(base.all_slots, window, update);
    if (!base.consistent_slots) return;
    if (const auto slot = storage_access_detail::StaticSlot(window)) {
      UpdateBucket(base.slots[*slot], window, update);
    } else {
      UpdateBucket(base.wildcard_slot, window, update);
    }
  }

  template <typename Query>
  [[nodiscard]] bool Any(const MemRefPtr& window, const Query& query) const {
    if (window && window->size_ == 0) return false;
    if (!window || !window->base_) return query(all_);
    if (query(opaque_)) return true;
    if (const auto found = bases_.find(window->base_.get()); found != bases_.end()) {
      const auto& base = found->second;
      const auto slot = storage_access_detail::StaticSlot(window);
      if (base.consistent_slots && base.slot_count == window->slot_count_ && slot) {
        if (const auto entry = base.slots.find(*slot); entry != base.slots.end()) {
          if (QueryBucket(entry->second, window, query)) return true;
        }
        return QueryBucket(base.wildcard_slot, window, query);
      }
      return QueryBucket(base.all_slots, window, query);
    }
    return false;
  }

  // Freeze temporal payloads after all spatial routing has been constructed.
  // Subsequent activations must use windows already registered by that build.
  template <typename Visit>
  void ForEachPayload(const Visit& visit) {
    visit(all_);
    visit(opaque_);
    const auto bucket = [&](Bucket& value) {
      visit(value.all);
      visit(value.opaque);
    };
    for (auto& [key, base] : bases_) {
      (void)key;
      bucket(base.all_slots);
      bucket(base.wildcard_slot);
      for (auto& [slot, value] : base.slots) {
        (void)slot;
        bucket(value);
      }
    }
    for (size_t i = 1; i < nodes_.size(); ++i) {
      visit(nodes_[i].all);
      visit(nodes_[i].cover);
    }
  }

 private:
  using Offset = storage_access_detail::Offset;
  using ByteRange = storage_access_detail::ByteRange;
  static constexpr Offset kOffsetBias = -(static_cast<Offset>(std::numeric_limits<int64_t>::min()));

  struct Node {
    size_t left = 0;
    size_t right = 0;
    Payload cover;
    Payload all;
  };
  struct Bucket {
    size_t root = 0;
    Offset begin = 0;
    Offset width = 0;
    Payload opaque;
    Payload all;
  };
  struct Base {
    VarPtr owner;
    uint64_t slot_count = 0;
    bool consistent_slots = false;
    Bucket all_slots;
    Bucket wildcard_slot;
    std::map<uint64_t, Bucket> slots;
  };

  static std::optional<ByteRange> Bytes(const MemRefPtr& window) {
    auto bytes = storage_access_detail::StaticBytes(window);
    if (bytes) {
      // Map signed offsets onto the nonnegative 65-bit tree domain.
      bytes->begin += kOffsetBias;
      bytes->end += kOffsetBias;
    }
    return bytes;
  }

  size_t NewNode() {
    nodes_.emplace_back();
    return nodes_.size() - 1;
  }

  template <typename Update>
  void UpdateBucket(Bucket& bucket, const MemRefPtr& window, const Update& update) {
    update(bucket.all);
    const auto bytes = Bytes(window);
    if (!bytes) {
      update(bucket.opaque);
      return;
    }
    if (bucket.root == 0) {
      bucket.width = 1;
      while (bytes->begin / bucket.width != (bytes->end - 1) / bucket.width) bucket.width *= 2;
      bucket.begin = (bytes->begin / bucket.width) * bucket.width;
      bucket.root = NewNode();
    }
    while (bytes->begin < bucket.begin || bucket.begin + bucket.width < bytes->end) {
      const size_t parent = NewNode();
      const Offset begin = (bucket.begin / (2 * bucket.width)) * (2 * bucket.width);
      if (begin == bucket.begin) {
        nodes_[parent].left = bucket.root;
      } else {
        nodes_[parent].right = bucket.root;
      }
      nodes_[parent].all = nodes_[bucket.root].all;
      bucket.root = parent;
      bucket.begin = begin;
      bucket.width *= 2;
    }
    UpdateNode(bucket.root, bucket.begin, bucket.begin + bucket.width, *bytes, update);
  }

  template <typename Update>
  void UpdateNode(size_t node, Offset begin, Offset end, ByteRange range, const Update& update) {
    update(nodes_[node].all);
    if (range.begin <= begin && end <= range.end) {
      update(nodes_[node].cover);
      return;
    }
    const Offset middle = begin + (end - begin) / 2;
    if (range.begin < middle) {
      if (nodes_[node].left == 0) {
        const size_t child = NewNode();
        nodes_[node].left = child;
      }
      UpdateNode(nodes_[node].left, begin, middle, range, update);
    }
    if (middle < range.end) {
      if (nodes_[node].right == 0) {
        const size_t child = NewNode();
        nodes_[node].right = child;
      }
      UpdateNode(nodes_[node].right, middle, end, range, update);
    }
  }

  template <typename Query>
  [[nodiscard]] bool QueryNode(size_t node, Offset begin, Offset end, ByteRange range,
                               const Query& query) const {
    if (node == 0 || range.end <= begin || end <= range.begin) return false;
    if (range.begin <= begin && end <= range.end) return query(nodes_[node].all);
    if (query(nodes_[node].cover)) return true;
    const Offset middle = begin + (end - begin) / 2;
    return (range.begin < middle && QueryNode(nodes_[node].left, begin, middle, range, query)) ||
           (middle < range.end && QueryNode(nodes_[node].right, middle, end, range, query));
  }

  template <typename Query>
  [[nodiscard]] bool QueryBucket(const Bucket& bucket, const MemRefPtr& window, const Query& query) const {
    if (bucket.root == 0) return query(bucket.all);
    const auto bytes = Bytes(window);
    if (!bytes) return query(bucket.all);
    return query(bucket.opaque) ||
           QueryNode(bucket.root, bucket.begin, bucket.begin + bucket.width, *bytes, query);
  }

  std::map<const Var*, Base> bases_;
  std::vector<Node> nodes_{Node{}};
  Payload opaque_;
  Payload all_;
};

/// Monotone observations of storage selected during planning. Each query uses
/// conservative maxima in both branch orders. Sparse fixed-width routing takes
/// O(log N) allocation/slot lookup work and O(N) space across N observations.
class StorageObserverIndex {
 public:
  void Clear() { windows_.Clear(); }

  void Observe(const MemRefPtr& window, size_t primary, size_t reverse) {
    windows_.UpdateWindow(window, [&](Summary& summary) {
      summary.primary = std::max(summary.primary, primary);
      summary.reverse = std::max(summary.reverse, reverse);
    });
  }

  [[nodiscard]] bool MayObserveAfter(const MemRefPtr& window, size_t primary, size_t reverse) const {
    return windows_.Any(window, [&](const Summary& summary) {
      return summary.primary > primary && summary.reverse > reverse;
    });
  }

 private:
  struct Summary {
    size_t primary = 0;
    size_t reverse = 0;
  };
  StorageWindowIndex<Summary> windows_;
};

}  // namespace pypto::ir

#endif  // PYPTO_IR_TRANSFORMS_UTILS_STORAGE_ACCESS_INDEX_H_
