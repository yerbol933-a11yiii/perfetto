/*
 * Copyright (C) 2026 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/trace_processor/core/exec/interval_fill_gaps.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "perfetto/base/logging.h"
#include "perfetto/base/status.h"
#include "perfetto/ext/base/status_macros.h"
#include "perfetto/ext/base/utils.h"
#include "src/trace_processor/core/common/storage_types.h"
#include "src/trace_processor/core/exec/column_view.h"
#include "src/trace_processor/core/exec/context.h"
#include "src/trace_processor/core/exec/group_table.h"
#include "src/trace_processor/core/exec/row_store.h"
#include "src/trace_processor/core/exec/selection.h"
#include "src/trace_processor/core/exec/variant.h"
#include "src/trace_processor/core/util/span.h"

namespace perfetto::trace_processor::core::exec {
namespace {

constexpr int64_t kEndOfTime = std::numeric_limits<int64_t>::max();

// The interval of a background row.
struct BackgroundSpan {
  int64_t start;
  int64_t end;

  // The row's index in `background_rows`.
  uint32_t row;
};

// A gap to output.
struct Filler {
  int64_t start;
  int64_t end;

  // The background row the gap came from.
  uint32_t background_row;

  // The lane the gap is in.
  uint32_t lane;
};

// Adds a null column of the same kind and type as `like`, or of `fallback` if
// `like` is not known.
void AddNulls(const std::optional<ColumnShape>& like,
              StorageType fallback,
              Context& context,
              RowBatch* out) {
  if (like && like->kind == ColumnView::Kind::kVariant) {
    // Zeroed, the null buffer's values are null Variants.
    const ColumnBuffer& nulls = context.NullBuffer();
    out->AddColumn(ColumnView::Variants(nulls.chunk().Values<Variant>()),
                   nulls);
    return;
  }
  out->AddNullColumn(like ? like->type : fallback, context);
}

}  // namespace

struct IntervalFillGaps::State : OperatorState {
  State() : OperatorState(ResetEachRun{}) {}
  ~State() override;
  void Reset() override;

  const Source* background = nullptr;
  Context* context = nullptr;
  std::unique_ptr<OperatorState> background_state;
  base::Status status = base::OkStatus();
  bool background_read = false;
  bool finished = false;

  // Maps PER values to lane numbers. Background lanes are added first.
  GroupTable lanes;

  // Background intervals, grouped by lane and sorted by ts.
  std::vector<BackgroundSpan> spans;

  // If `background_has_lanes_`, the index in `spans` where each lane starts.
  // Has a final entry for the end.
  std::vector<uint32_t> lane_spans;

  // If `background_has_lanes_`, whether each background lane had any input.
  std::vector<bool> visited;

  // All background rows. Fillers take their columns from these.
  RowStore background_rows;

  // The number of input groups seen.
  uint32_t input_groups = 0;

  // Whether a lane is being read, and its number.
  bool in_lane = false;
  uint32_t lane = 0;

  // The end of the current lane's input seen so far.
  int64_t covered = 0;

  // The current lane's background intervals not yet processed, as a range
  // of `spans`.
  uint32_t next_span = 0;
  uint32_t end_span = 0;

  // The shape of each output's input column, from the first input batch.
  std::vector<std::optional<ColumnShape>> shapes;

  // The type of each PER column, from the first input batch.
  std::vector<StorageType> key_types;

  // Gaps not yet output, and how many of them have been output.
  std::vector<Filler> fillers;
  size_t served = 0;

  // Whether the current input batch has been output but its gaps have not.
  bool serving = false;

  // Buffers reused across calls; nothing in them outlives the next call.
  RowBatch batch;
  RowBatch starts;
  std::vector<uint32_t> start_rows;
  std::vector<uint32_t> groups;
  std::vector<uint32_t> rows;
  RowBatch gathered;
};

IntervalFillGaps::State::~State() = default;

void IntervalFillGaps::State::Reset() {
  background->Rewind(*background_state);
  status = base::OkStatus();
  background_read = false;
  finished = false;
  lanes.Clear();
  spans.clear();
  lane_spans.clear();
  visited.clear();
  background_rows.Clear();
  in_lane = false;
  input_groups = 0;
  std::fill(shapes.begin(), shapes.end(), std::nullopt);
  key_types.clear();
  fillers.clear();
  served = 0;
  serving = false;
}

IntervalFillGaps::IntervalFillGaps(IntervalFillGapsSpec spec)
    : spec_(std::move(spec)) {
  background_has_lanes_ =
      spec_.background_key_columns.size() == spec_.key_columns.size();
  PERFETTO_CHECK(background_has_lanes_ || spec_.background_key_columns.empty());
  for (const IntervalFillGapsSpec::Output& output : spec_.outputs) {
    input_columns_.push_back(output.input);
  }
}

IntervalFillGaps::~IntervalFillGaps() = default;

std::unique_ptr<OperatorState> IntervalFillGaps::MakeState(
    Context& context) const {
  auto state = std::make_unique<State>();
  state->background = spec_.background;
  state->context = &context;
  state->background_state = spec_.background->MakeState(context);
  state->shapes.resize(spec_.outputs.size());
  return state;
}

base::Status IntervalFillGaps::status(const OperatorState& state) const {
  return state.Cast<const State>().status;
}

bool IntervalFillGaps::EnsureBackgroundRead(State& s) const {
  if (!s.background_read) {
    s.status = ReadBackground(s);
    s.background_read = true;
  }
  return s.status.ok();
}

base::Status IntervalFillGaps::ReadBackground(State& s) const {
  RowBatch& batch = s.batch;
  bool named = background_has_lanes_ && !spec_.key_columns.empty();
  while (spec_.background->GetData(batch, *s.background_state)) {
    PERFETTO_DCHECK(IsFlat<Int64>(batch.column(spec_.background_ts_column)));
    PERFETTO_DCHECK(IsFlat<Int64>(batch.column(spec_.background_dur_column)));
    if (named) {
      s.groups.resize(batch.size());
      if (std::optional<uint32_t> bad = s.lanes.FindOrCreate(
              batch, spec_.background_key_columns, s.groups.data())) {
        return base::ErrStatus(
            "INTERVAL FILL GAPS: PER column %u of the background cannot be a "
            "key",
            *bad + 1);
      }
    }
    FlatColumnReader<int64_t> ts(batch, spec_.background_ts_column);
    FlatColumnReader<int64_t> dur(batch, spec_.background_dur_column);
    uint32_t first_row = s.background_rows.size();
    for (uint32_t row = 0; row < batch.size(); ++row) {
      if (named && s.groups[row] + 1 > s.lane_spans.size()) {
        // Rows are grouped, so a new lane always gets the next number.
        PERFETTO_DCHECK(s.groups[row] == s.lane_spans.size());
        s.lane_spans.push_back(static_cast<uint32_t>(s.spans.size()));
      }
      int64_t start;
      int64_t length;
      if (!ts.Read(row, &start) || !dur.Read(row, &length)) {
        continue;
      }
      if (start < 0 || length < 0) {
        return base::ErrStatus(
            "INTERVAL FILL GAPS: a background row's ts or dur is below zero");
      }
      int64_t end;
      if (!base::CheckedAdd(start, length, &end)) {
        return base::ErrStatus("INTERVAL FILL GAPS: integer overflow");
      }
      if (length == 0) {
        continue;
      }
      // Rows are sorted by ts, so only the previous interval in the same lane
      // can overlap.
      bool same_lane = !named || s.spans.size() > s.lane_spans.back();
      if (same_lane && !s.spans.empty()) {
        PERFETTO_DCHECK(start >= s.spans.back().start);
        if (start < s.spans.back().end) {
          return base::ErrStatus(
              "INTERVAL FILL GAPS: background rows overlap within a lane");
        }
      }
      s.spans.push_back({start, end, first_row + row});
    }
    // Keep every row, even ones without an interval, so row indexes match.
    RETURN_IF_ERROR(s.background_rows.Append(batch));
  }
  RETURN_IF_ERROR(spec_.background->status(*s.background_state));
  if (background_has_lanes_) {
    if (spec_.key_columns.empty()) {
      s.lane_spans.push_back(0);
    }
    s.lane_spans.push_back(static_cast<uint32_t>(s.spans.size()));
    s.visited.resize(s.lane_spans.size() - 1);
  }
  return base::OkStatus();
}

void IntervalFillGaps::StartLane(uint32_t lane, State& s) const {
  if (s.in_lane) {
    EndLane(s);
  }
  s.in_lane = true;
  s.lane = lane;
  s.covered = 0;
  if (!background_has_lanes_) {
    s.next_span = 0;
    s.end_span = static_cast<uint32_t>(s.spans.size());
  } else if (lane < s.visited.size()) {
    s.visited[lane] = true;
    s.next_span = s.lane_spans[lane];
    s.end_span = s.lane_spans[lane + 1];
  } else {
    // The background has no rows for this lane, so there is nothing to fill.
    s.next_span = s.end_span = 0;
  }
}

void IntervalFillGaps::EndLane(State& s) const {
  FillTo(kEndOfTime, s);
  s.in_lane = false;
}

void IntervalFillGaps::FillTo(int64_t to, State& s) const {
  // Rows are sorted by ts, so gaps before `covered` were already recorded.
  // Everything between `covered` and `to` is a gap.
  for (; s.next_span < s.end_span; ++s.next_span) {
    const BackgroundSpan& span = s.spans[s.next_span];
    if (span.start >= to) {
      return;
    }
    int64_t start = std::max(span.start, s.covered);
    int64_t end = std::min(span.end, to);
    if (start < end) {
      s.fillers.push_back({start, end, span.row, s.lane});
    }
    if (span.end > to) {
      return;
    }
  }
}

base::Status IntervalFillGaps::Cover(const RowBatch& in, State& s) const {
  PERFETTO_DCHECK(IsFlat<Int64>(in.column(spec_.ts_column)));
  PERFETTO_DCHECK(IsFlat<Int64>(in.column(spec_.dur_column)));

  // Find the rows that start a new lane, and look up their lane numbers.
  // Only these rows are looked up, since the rest of a lane has the same
  // keys.
  s.start_rows.clear();
  if (spec_.key_columns.empty()) {
    if (!s.in_lane && in.size() > 0) {
      s.start_rows.push_back(0);
      s.groups.assign(1, 0);
    }
  } else {
    if (s.key_types.empty()) {
      for (uint32_t column : spec_.key_columns) {
        s.key_types.push_back(KeyType(in.column(column)));
      }
    }
    FlatColumnReader<uint32_t> group(in, spec_.group_column);
    for (uint32_t row = 0; row < in.size(); ++row) {
      uint32_t number;
      PERFETTO_CHECK(group.Read(row, &number));
      if (number == s.input_groups) {
        s.start_rows.push_back(row);
        ++s.input_groups;
      }
    }
    if (!s.start_rows.empty()) {
      s.starts.CopyFrom(in);
      const uint32_t* rows = s.start_rows.data();
      s.starts.mutable_selection().Keep({rows, rows + s.start_rows.size()});
      s.groups.resize(s.start_rows.size());
      if (std::optional<uint32_t> bad = s.lanes.FindOrCreate(
              s.starts, spec_.key_columns, s.groups.data())) {
        return base::ErrStatus(
            "INTERVAL FILL GAPS: PER column %u must hold one type, the same in "
            "the input as in the background",
            *bad + 1);
      }
    }
  }

  FlatColumnReader<int64_t> ts(in, spec_.ts_column);
  FlatColumnReader<int64_t> dur(in, spec_.dur_column);
  size_t next_start = 0;
  for (uint32_t row = 0; row < in.size(); ++row) {
    if (next_start < s.start_rows.size() && s.start_rows[next_start] == row) {
      StartLane(s.groups[next_start++], s);
    }
    int64_t start;
    int64_t length;
    if (!ts.Read(row, &start) || !dur.Read(row, &length)) {
      continue;
    }
    if (start < 0 || length < -1) {
      return base::ErrStatus(
          "INTERVAL FILL GAPS: a row's ts is below zero, or its dur is below "
          "zero and not -1");
    }
    int64_t end = kEndOfTime;
    if (length >= 0 && !base::CheckedAdd(start, length, &end)) {
      return base::ErrStatus("INTERVAL FILL GAPS: integer overflow");
    }
    if (length != 0) {
      FillTo(start, s);
      s.covered = std::max(s.covered, end);
    }
  }
  return base::OkStatus();
}

OpResult IntervalFillGaps::Execute(const RowBatch& in,
                                   RowBatch& out,
                                   OperatorState& state) const {
  auto& s = state.Cast<State>();
  out.Reset();
  if (s.serving) {
    return ServeFillers(out, s);
  }
  // Background lanes must be known before reading input.
  if (!EnsureBackgroundRead(s)) {
    return OpResult::kError;
  }
  s.status = Cover(in, s);
  if (!s.status.ok()) {
    return OpResult::kError;
  }
  for (size_t i = 0; i < spec_.outputs.size(); ++i) {
    if (!s.shapes[i]) {
      s.shapes[i] = ColumnShape::Of(in.column(spec_.outputs[i].input));
    }
  }
  out.Project(in, input_columns_);
  s.serving = !s.fillers.empty();
  return s.serving ? OpResult::kHaveMoreOutput : OpResult::kNeedMoreInput;
}

OpResult IntervalFillGaps::Finish(RowBatch& out, OperatorState& state) const {
  auto& s = state.Cast<State>();
  out.Reset();
  if (!EnsureBackgroundRead(s)) {
    return OpResult::kError;
  }
  if (!s.finished) {
    if (s.in_lane) {
      EndLane(s);
    }
    // Background lanes with no input are filled completely.
    for (uint32_t lane = 0; lane < s.visited.size(); ++lane) {
      if (!s.visited[lane]) {
        StartLane(lane, s);
        EndLane(s);
      }
    }
    s.finished = true;
  }
  if (s.fillers.empty()) {
    return OpResult::kNeedMoreInput;
  }
  return ServeFillers(out, s);
}

OpResult IntervalFillGaps::ServeFillers(RowBatch& out, State& s) const {
  auto count = static_cast<uint32_t>(
      std::min<size_t>(s.fillers.size() - s.served, kMaxBatchRows));
  const Filler* fillers = s.fillers.data() + s.served;

  // Write the fillers' ts and dur, and gather their background rows.
  ColumnBuffer ts_buffer = s.context->TakeBuffer();
  ColumnBuffer dur_buffer = s.context->TakeBuffer();
  int64_t* ts = ts_buffer.chunk().Values<int64_t>();
  int64_t* dur = dur_buffer.chunk().Values<int64_t>();
  s.rows.resize(count);
  for (uint32_t i = 0; i < count; ++i) {
    ts[i] = fillers[i].start;
    dur[i] = fillers[i].end - fillers[i].start;
    s.rows[i] = fillers[i].background_row;
  }
  s.background_rows.View(&s.gathered, {s.rows.data(), s.rows.data() + count},
                         *s.context);
  bool lane_values = !background_has_lanes_;
  if (lane_values) {
    for (uint32_t i = 0; i < count; ++i) {
      s.rows[i] = fillers[i].lane;
    }
  }

  // ts and dur always come from the filler, not the background.
  for (size_t i = 0; i < spec_.outputs.size(); ++i) {
    const IntervalFillGapsSpec::Output& output = spec_.outputs[i];
    auto key = std::find(spec_.key_columns.begin(), spec_.key_columns.end(),
                         output.input);
    if (output.input == spec_.ts_column) {
      out.AddColumn(ColumnView::Reference(StorageType{Int64{}}, ts), ts_buffer);
    } else if (output.input == spec_.dur_column) {
      out.AddColumn(ColumnView::Reference(StorageType{Int64{}}, dur),
                    dur_buffer);
    } else if (output.background) {
      const ColumnView& column = s.gathered.column(*output.background);
      if (s.shapes[i] &&
          !SameLogicalType(ColumnShape::Of(column), *s.shapes[i])) {
        s.status = base::ErrStatus(
            "INTERVAL FILL GAPS: column '%s' holds a different type in the "
            "background than in the input",
            output.name.c_str());
        return OpResult::kError;
      }
      out.AddColumn(column, s.gathered.buffer(*output.background));
    } else if (lane_values && key != spec_.key_columns.end()) {
      auto k = static_cast<uint32_t>(key - spec_.key_columns.begin());
      ColumnBuffer buffer = s.context->TakeBuffer();
      ColumnView view =
          WriteKeys(s.lanes, k, s.rows.data(), count, s.key_types[k], buffer);
      out.AddColumn(view, std::move(buffer));
    } else {
      AddNulls(s.shapes[i], output.fallback, *s.context, &out);
    }
  }
  out.SetRowCount(count);
  s.served += count;
  if (s.served < s.fillers.size()) {
    return OpResult::kHaveMoreOutput;
  }
  s.fillers.clear();
  s.served = 0;
  s.serving = false;
  return OpResult::kNeedMoreInput;
}

}  // namespace perfetto::trace_processor::core::exec
