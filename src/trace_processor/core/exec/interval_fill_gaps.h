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

#ifndef SRC_TRACE_PROCESSOR_CORE_EXEC_INTERVAL_FILL_GAPS_H_
#define SRC_TRACE_PROCESSOR_CORE_EXEC_INTERVAL_FILL_GAPS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "perfetto/base/status.h"
#include "src/trace_processor/core/common/storage_types.h"
#include "src/trace_processor/core/exec/column_view.h"
#include "src/trace_processor/core/exec/operator.h"
#include "src/trace_processor/core/exec/row_batch.h"

namespace perfetto::trace_processor::core::exec {

struct IntervalFillGapsSpec {
  // An output column. Input rows take its value from the input. Filler rows
  // take it from the first of these that applies:
  //  - the background column with the same name;
  //  - the lane's PER value, if it is a PER column;
  //  - null.
  struct Output {
    // Column index in the input.
    uint32_t input = 0;
    // Column index in the background, if it has this column.
    std::optional<uint32_t> background;
    // Type of the null column used if no input batch was seen.
    StorageType fallback{Int64{}};
    // Column name, used in error messages.
    std::string name;
  };

  // Input columns. ts and dur are Int64. Rows are grouped by `key_columns`
  // and sorted by ts within each group.
  uint32_t ts_column = 0;
  uint32_t dur_column = 0;
  std::vector<uint32_t> key_columns;

  // The input column holding each row's group number. Only used if there
  // are key columns.
  uint32_t group_column = 0;

  // Background columns. ts and dur are Int64. Rows are grouped by
  // `background_key_columns` and sorted by ts within each group.
  const Source* background = nullptr;
  uint32_t background_ts_column = 0;
  uint32_t background_dur_column = 0;

  // Either empty, or the background columns matching `key_columns`, in the
  // same order.
  std::vector<uint32_t> background_key_columns;

  std::vector<Output> outputs;
};

// INTERVAL FILL GAPS. Outputs every input row unchanged. Also outputs a
// filler row for each part of the background that no input row in the same
// lane covers.
//
// A lane is the set of rows with the same PER values. There are two modes:
//  - The background has the PER columns, or there are no PER columns. Each
//    background row belongs to one lane. A lane with no input is filled
//    completely. Input in a lane the background does not have gets no
//    fillers.
//  - The background has no PER columns but the input does. Every input lane
//    is filled against the whole background. Fillers get the lane's PER
//    values.
//
// Rules for intervals:
//  - An input row with a null ts or dur covers nothing.
//  - A dur of -1 means the row lasts forever.
//  - A negative ts is an error on both sides.
//  - Input rows may overlap. Background rows in the same lane may not.
//  - Fillers never have zero duration.
// Output rows are in no particular order.
//
// Input arrives one lane at a time, sorted by ts, so gaps are found while
// streaming. When a row starts after the end of all earlier rows in its lane,
// the background between them is a gap.
class IntervalFillGaps : public Operator {
 public:
  static constexpr char kType = 0;
  const void* type() const override { return &kType; }

  explicit IntervalFillGaps(IntervalFillGapsSpec);
  ~IntervalFillGaps() override;

  std::unique_ptr<OperatorState> MakeState(Context&) const override;
  OpResult Execute(const RowBatch&, RowBatch&, OperatorState&) const override;
  OpResult Finish(RowBatch&, OperatorState&) const override;
  base::Status status(const OperatorState&) const override;

 private:
  friend class PipelineFormatter;

  struct State;

  // Reads the background on first call. Returns false on error.
  bool EnsureBackgroundRead(State&) const;
  base::Status ReadBackground(State&) const;

  // Finds the gaps left by the rows of one input batch.
  base::Status Cover(const RowBatch&, State&) const;

  // Starts lane `lane`. Ends the current lane first, if there is one.
  void StartLane(uint32_t lane, State&) const;

  // Ends the current lane. All of its remaining background is a gap.
  void EndLane(State&) const;

  // Records the gaps in the current lane's background before `to`.
  void FillTo(int64_t to, State&) const;

  // Writes the next batch of filler rows to `out`.
  OpResult ServeFillers(RowBatch& out, State&) const;

  IntervalFillGapsSpec spec_;

  // Whether each background row belongs to one lane. If not, every lane
  // shares the whole background.
  bool background_has_lanes_;

  // The input column of each output.
  std::vector<uint32_t> input_columns_;
};

}  // namespace perfetto::trace_processor::core::exec

#endif  // SRC_TRACE_PROCESSOR_CORE_EXEC_INTERVAL_FILL_GAPS_H_
