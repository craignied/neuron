#!/usr/bin/env python3
"""Orchestrate optimizer_probe: interleaving, validation, comparison groups.

Bare python3 -- no pip packages, consistent with every other tool here.

WHY AN ORCHESTRATOR: the probe runs arms in the order it is given, and a fixed
order lets machine drift (thermal, background load, cache state) accumulate
against whichever arm always runs last.  This shuffles arm order independently
within every repetition from a recorded seed, so the order is reproducible but
never systematic.

WHAT IT REFUSES TO DO.  The orchestration layer is part of the evidence system,
not a pretty printer, so a result set that cannot support a conclusion is
REFUSED rather than warned about:

  * an unusable arm is never averaged into a timing -- it is counted and named;
  * a comparison group whose arms disagree on anything except the group's
    declared axis is refused, because those timings are not comparable;
  * a comparison group whose arms do not share a proven parameter-state
    identity is refused;
  * rows from a different binary, schema, or source tree are refused;
  * a probe that fails to emit exactly one well-formed row is refused.

  ./tests/optimizer/run_probe.py --pilot
  ./tests/optimizer/run_probe.py --reps 15 --seed 20260804 --out results.jsonl
  ./tests/optimizer/run_probe.py --self-test
"""

import argparse
import copy
import json
import os
import random
import shutil
import subprocess
import sys
import time

SCHEMA_EXPECTED = 3

# The identity fields a row must share with the binary that produced it.
# source_id is the authority: it covers the harness AND the engine sources AND
# the build file, so an edit to src/network.cpp moves it.  source_files is
# carried alongside because a changed source SET -- a file added or removed --
# is a different kind of difference from a changed source file, and a reader
# should be able to tell them apart.
IDENTITY_FIELDS = ["rev", "dirty", "source_id", "source_files", "engine_id",
                   "engine_files", "build"]

REQUIRED_FIELDS = [
    "schema", "rev", "dirty", "source_id", "source_files", "engine_id",
    "engine_files", "build", "case",
    "comparison_group",
    "group_axis", "fixture", "split", "data_id", "data_seed", "test_fraction",
    "endpoint", "timing_scope", "workload", "cv_folds", "cv_repeats",
    "model", "arch", "loss", "rows", "rows_total", "rows_test", "inputs",
    "params", "weight_seed", "weight_id_available", "weight_start_id",
    "weight_end_id", "weight_elements", "weight_id_note", "function_start_id",
    "function_end_id", "optimizer", "lbfgs_memory", "method", "optimizer_name",
    "mode", "eta",
    "auto_step", "decay_on", "decay", "grad_stop", "auto_stop", "min_stop",
    "target", "achieved",
    "ceiling", "iteration_index", "iterations_completed", "full_passes",
    "elapsed_ns", "peak_rss_kb", "stop_reason", "heldout_error", "cv_auc",
    "locked_auc", "cv_folds_ok", "cv_folds_total", "converged", "target_reached",
    "finite", "usable", "failure_stage", "error",
]

# auto_step and auto_stop are DIFFERENT ENGINE FEATURES with confusingly close
# names, and both are real: setAutoStepSize is the per-iteration step-size
# search, setAutoStop is the plateau stopping rule.  Both are group invariants,
# so an arm cannot differ from its group in either without being refused.
BOOL_FIELDS = ["dirty", "weight_id_available", "auto_step", "decay_on",
               "grad_stop", "auto_stop", "min_stop",
               "converged", "target_reached", "finite", "usable"]

STR_FIELDS = ["rev", "source_id", "engine_id", "build", "case",
              "comparison_group",
              "group_axis", "fixture", "split", "data_id", "endpoint",
              "timing_scope", "workload", "model", "arch", "loss",
              "weight_start_id", "weight_end_id", "weight_id_note",
              "function_start_id", "function_end_id", "method",
              "optimizer_name", "mode",
              "stop_reason", "failure_stage", "error"]

# Non-negative integers.  bool is a SUBCLASS of int in Python, so True would
# satisfy a naive isinstance(x, int) check and slip through as 1 -- every
# integer check below rejects bool explicitly.
INT_FIELDS = ["schema", "rows", "rows_total", "rows_test", "inputs", "params",
              "weight_seed", "weight_elements", "optimizer", "lbfgs_memory",
              "ceiling",
              "iteration_index", "elapsed_ns", "source_files", "engine_files",
              "data_seed",
              "cv_folds", "cv_repeats", "cv_folds_ok", "cv_folds_total"]

# Real numbers that must be present and finite.
NUM_FIELDS = ["eta", "decay", "target", "test_fraction"]

# Integers that MAY be null, and what null means for each.
NULLABLE_INT_FIELDS = ["iterations_completed", "full_passes", "peak_rss_kb"]

# Closed domains, checked so a row cannot report a configuration the engine
# would refuse.  These mirror the C++ validate(); a row that disagrees with them
# came from a binary that is not the one this runner understands.
DOMAINS = {
    "eta": (0.0, 1.0),
    "decay": (0.0, 1.0),
    "target": (0.0, 1.0),       # exclusive both ends; checked separately
    "test_fraction": (0.0, 1.0),
}

ENDPOINTS = {"practical", "strict", "none"}
SCOPES = {"optimizer", "workflow"}
WORKLOADS = {"fit", "cv"}

MODELS = {"logistic", "simpleprop", "bareprop", "backprop"}
MODES = {"batch", "online"}
LOSSES = {"xentropy", "lms"}
STAGES = {"none", "refused", "setup", "training"}
STOP_REASONS = {"none", "max_iterations", "min_error", "min_change",
                "error_window", "grad_max", "plateau", "cancelled",
                "validation_early_stop", "probe_budget",
                # Not engine stop reasons: a cv arm ran k*r+1 fits and has no
                # single one.  Named distinctly so nothing can read a workflow
                # outcome as an Iterative::StopReason.
                "cv_complete", "cv_incomplete"}

# Real numbers that MAY be null or -1 ("not applicable here").
NULLABLE_NUM_FIELDS = ["heldout_error", "cv_auc", "locked_auc"]

# 3 is the RESEARCH-ONLY L-BFGS prototype (Network::TRAIN_LBFGS).  No menu, GUI
# control, HTTP field or automatic-selection rule produces it; it exists so this
# harness and tests/network/check_lbfgs.cpp can measure the Phase 3 candidate.
OPTIMIZER_NAMES = {0: "canonical", 1: "cgd", 2: "shanno", 3: "lbfgs"}

# Fields that must be identical across every arm of a comparison group.  The
# group's declared axis is removed from this list for that group, and nothing
# else may vary: two arms differing in eta, loss, ceiling or branch are not an
# optimizer comparison however similar they look.
GROUP_INVARIANTS = [
    "split", "data_id", "data_seed", "test_fraction", "fixture", "model",
    "arch", "loss", "rows", "rows_total", "rows_test", "inputs", "params",
    "weight_seed", "weight_start_id", "mode", "eta", "auto_step", "decay_on",
    "decay", "grad_stop", "auto_stop", "min_stop", "target", "ceiling",
    # L-BFGS's memory length. An invariant for the same reason as the rest: two
    # arms at different m are two methods, so a group holding both is varying
    # something it has not declared.  The memory-sweep group declares
    # `lbfgs_memory` as its axis, which removes it from this list there.
    "lbfgs_memory",
    # THE TWO STEP 0B ADDITIONS, and the reason they are invariants rather than
    # descriptive labels.  `endpoint` pins WHICH TARGET the arms raced to: a
    # practical arm and a strict arm are aimed at different objectives, so a
    # group holding both would report a speed ratio between two different
    # races.  `timing_scope` pins WHAT THE CLOCK COVERED: an optimizer-only
    # number against a whole-workflow one is a comparison between two different
    # jobs, and it is precisely the mislabelling Step 0B must not produce.
    # `workload`, `cv_folds` and `cv_repeats` complete that: five folds twice
    # is not five folds once.
    "endpoint", "timing_scope", "workload", "cv_folds", "cv_repeats",
]

# What each declared axis is ALLOWED to vary -- a set, because "which method"
# is not one field.  "canonical fixed eta" and "canonical automatic step size"
# share a trainingType and are not the same method: the search costs maxLoops
# extra full passes per iteration.  So the `method` axis frees auto_step along
# with the optimizer, and nothing else.
AXIS_FIELDS = {
    "optimizer": {"optimizer"},
    "method": {"optimizer", "auto_step"},
    "auto_step": {"auto_step"},
    "grad_stop branch": {"grad_stop"},
    # The L-BFGS memory length, and ONLY it: a group on this axis holds one
    # optimizer at several m, so `optimizer` stays invariant there.  Declaring
    # the axis is what permits lbfgs_memory to differ; every other field still
    # has to match.
    "lbfgs_memory": {"lbfgs_memory"},
    "none": set(),
}


# THE REPETITION POLICY, scaled to run cost rather than applied uniformly.
#
# Fifteen interleaved repetitions is right for a millisecond cell and wasteful
# for a four-minute one: an hour of duplicated compute does not make a stable
# 10x effect more true.  So the ladder below picks the count from the WARM-UP's
# own measured duration -- a number the campaign already pays for and discards.
#
# Every choice is RECORDED with its reason and the observation behind it, in the
# campaign record and in the summary, because "we ran this one 3 times" is a
# fact a reader must be able to see rather than infer.  Where the reduced count
# leaves a group's ordering ambiguous -- overlapping p10/p90 intervals between
# the two fastest arms -- that is reported too, rather than resolved silently.
REPETITION_LADDER = [
    (1.0, 15, "short cell: warm-up under 1 s"),
    (120.0, 5, "seconds-to-minutes cell: warm-up under 2 min"),
    (None, 3, "expensive cell: warm-up over 2 min, 15 reps would cost "
              "over half an hour for one arm"),
]


def reps_for(warmup_seconds, warmup_usable=True):
    """(count, reason) for a cell whose warm-up took this long.

    AN ARM THAT DID NOT REACH ITS ENDPOINT IS RUN ONCE.  It contributes no time
    to any summary -- the runner refuses to average it -- so repeating it buys
    no precision about anything.  It is not skipped either: one measured row is
    what puts the failure in the results, where a reader can see which method
    could not finish and why.  This is not a small saving: a failing arm burns
    its entire iteration ceiling on every run, which on the larger workloads is
    the most expensive thing in the campaign, spent to learn nothing.
    """
    if not warmup_usable:
        return 1, ("did not reach its endpoint at warm-up: a failed arm "
                   "contributes no timing, so it is recorded once rather than "
                   "repeated (warm-up %.3f s)" % warmup_seconds)
    for limit, count, why in REPETITION_LADDER:
        if limit is None or warmup_seconds < limit:
            return count, "%s (warm-up %.3f s)" % (why, warmup_seconds)
    raise AssertionError("the ladder must end in an open bucket")


class Refusal(Exception):
    """A result set that cannot support a conclusion."""


def refuse(msg):
    raise Refusal(msg)


# ---------------------------------------------------------------------------
# Row validation


def validate_row(row, expected_case=None, identity=None):
    """Validate one row in isolation.  Raises Refusal."""
    if not isinstance(row, dict):
        refuse("row is not a JSON object")

    missing = [f for f in REQUIRED_FIELDS if f not in row]
    if missing:
        refuse("row is missing fields: %s" % ", ".join(missing))

    if row["schema"] != SCHEMA_EXPECTED:
        refuse("row has schema %r, this runner expects %d"
               % (row["schema"], SCHEMA_EXPECTED))

    for f in BOOL_FIELDS:
        if not isinstance(row[f], bool):
            refuse("field %s must be a boolean, got %r" % (f, row[f]))
    for f in STR_FIELDS:
        if not isinstance(row[f], str):
            refuse("field %s must be a string, got %r" % (f, row[f]))

    # bool is a subclass of int, so True would pass a naive integer check and be
    # read as 1.  Rejected explicitly, everywhere an integer is expected.
    for f in INT_FIELDS:
        if isinstance(row[f], bool) or not isinstance(row[f], int):
            refuse("field %s must be an integer (not bool), got %r" % (f, row[f]))
        if row[f] < 0:
            refuse("field %s must be non-negative, got %r" % (f, row[f]))

    for f in NUM_FIELDS:
        v = row[f]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            refuse("field %s must be a number (not bool), got %r" % (f, v))
        if v != v or v in (float("inf"), float("-inf")):
            refuse("field %s must be finite, got %r" % (f, v))
        lo, hi = DOMAINS[f]
        if not (lo <= v <= hi):
            refuse("field %s is outside [%g,%g]: %r" % (f, lo, hi, v))
    if not (0.0 < row["target"] < 1.0):
        refuse("target must lie strictly inside (0,1): %r" % row["target"])

    # achieved is nullable: a non-finite objective has no JSON spelling, and the
    # `finite` flag is what tells a null apart from an absence.
    ach = row["achieved"]
    if ach is None:
        if row["finite"]:
            refuse("achieved is null but the row claims a finite objective")
    else:
        if isinstance(ach, bool) or not isinstance(ach, (int, float)):
            refuse("achieved must be a number or null, got %r" % ach)
        if ach != ach or ach in (float("inf"), float("-inf")):
            refuse("achieved is present but not finite: %r" % ach)
        if not row["finite"]:
            refuse("achieved is a finite number but the row says finite=false")

    for f in NULLABLE_INT_FIELDS:
        v = row[f]
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, int):
            refuse("field %s must be an integer or null (not bool), got %r" % (f, v))
        if v < 0:
            refuse("field %s must be non-negative, got %r" % (f, v))

    for f in NULLABLE_NUM_FIELDS:
        v = row[f]
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            refuse("field %s must be a number or null (not bool), got %r" % (f, v))
        if v != v or v in (float("inf"), float("-inf")):
            refuse("field %s is present but not finite: %r" % (f, v))

    if row["endpoint"] not in ENDPOINTS:
        refuse("unknown endpoint %r" % row["endpoint"])
    # AN ARM NOT RACING TO AN OBJECTIVE MUST NOT CLAIM AN OBJECTIVE ENDPOINT.
    # The cv arms stop on the engine's plateau rule because the matched
    # objective was measurably unreachable on 4 of 10 folds; that makes them a
    # fair comparison but not a matched-endpoint race, and the two must not be
    # readable as the same thing.
    if not row["min_stop"] and row["endpoint"] != "none":
        refuse("case %r arms no objective target but declares the %r endpoint"
               % (row["case"], row["endpoint"]))
    if not row["min_stop"] and not row["auto_stop"]:
        refuse("case %r arms no stopping rule that can fire" % row["case"])
    if row["timing_scope"] not in SCOPES:
        refuse("unknown timing_scope %r" % row["timing_scope"])
    if row["workload"] not in WORKLOADS:
        refuse("unknown workload %r" % row["workload"])

    # A cv arm's clock necessarily covers each fold's scoring epilogue, so it
    # cannot be an optimizer-only timing.  This is the mislabelling the whole
    # scope field exists to prevent, so it is refused rather than noted.
    if row["workload"] == "cv":
        if row["timing_scope"] != "workflow":
            refuse("case %r is a cv workload but claims %r scope; a cv clock "
                   "covers each fold's scoring epilogue and is never "
                   "optimizer-only" % (row["case"], row["timing_scope"]))
        if row["cv_folds"] < 2 or row["cv_repeats"] < 1:
            refuse("case %r is a cv workload with %r folds and %r repetitions"
                   % (row["case"], row["cv_folds"], row["cv_repeats"]))
    else:
        if row["cv_folds"] or row["cv_repeats"]:
            refuse("case %r is not a cv workload but reports %r folds / %r "
                   "repetitions" % (row["case"], row["cv_folds"],
                                    row["cv_repeats"]))
        if row["cv_folds_ok"] or row["cv_folds_total"]:
            refuse("case %r is not a cv workload but counted folds" % row["case"])
    if row["cv_folds_ok"] > row["cv_folds_total"]:
        refuse("case %r reports %r good folds of %r"
               % (row["case"], row["cv_folds_ok"], row["cv_folds_total"]))

    # THE METHOD NAME IS DERIVED, so it cannot disagree with what it names.
    expected_method = OPTIMIZER_NAMES.get(row["optimizer"], "?") + \
        ("-autostep" if row["auto_step"] else "")
    if row["method"] != expected_method:
        refuse("case %r reports method %r but optimizer %r with auto_step %r "
               "is %r" % (row["case"], row["method"], row["optimizer"],
                          row["auto_step"], expected_method))

    if row["optimizer"] not in OPTIMIZER_NAMES:
        refuse("optimizer %r is not a known training type" % row["optimizer"])
    if row["model"] not in MODELS:
        refuse("unknown model %r" % row["model"])
    if row["mode"] not in MODES:
        refuse("unknown mode %r" % row["mode"])
    if row["loss"] not in LOSSES:
        refuse("unknown loss %r" % row["loss"])
    if row["stop_reason"] not in STOP_REASONS:
        refuse("unknown stop_reason %r" % row["stop_reason"])
    if row["ceiling"] < 1:
        refuse("ceiling must be at least 1, got %r" % row["ceiling"])
    if not row["case"]:
        refuse("case name is empty")
    if not row["comparison_group"]:
        refuse("comparison_group is empty")

    if expected_case is not None and row["case"] != expected_case:
        refuse("asked for case %r but the row says %r"
               % (expected_case, row["case"]))

    if OPTIMIZER_NAMES.get(row["optimizer"]) != row["optimizer_name"]:
        refuse("optimizer %r does not match its name %r"
               % (row["optimizer"], row["optimizer_name"]))

    if identity is not None:
        for f in IDENTITY_FIELDS:
            if row[f] != identity[f]:
                refuse("row %s is %r but the invoked binary reports %r -- a "
                       "stale binary must not claim another source tree"
                       % (f, row[f], identity[f]))

    stage = row["failure_stage"]
    if stage not in ("none", "refused", "setup", "training"):
        refuse("unknown failure_stage %r" % stage)

    # An executed arm must carry real provenance.  A refused or setup-failed arm
    # legitimately has less, so the requirement is scoped to arms that ran.
    executed = stage in ("none", "training")
    if executed and not row["split"]:
        refuse("case %r executed but carries no split identity" % row["case"])
    if executed and not row["data_id"]:
        refuse("case %r executed but carries no data identity" % row["case"])
    if executed:
        # THE SPLIT MUST ADD UP.  A holdout that trained on more rows than the
        # dataset holds, or that reported a test set it never took, describes a
        # partition that did not happen.
        if row["rows"] + row["rows_test"] > row["rows_total"]:
            refuse("case %r trained on %r rows and held out %r, but the dataset "
                   "has %r" % (row["case"], row["rows"], row["rows_test"],
                               row["rows_total"]))
        if row["test_fraction"] > 0 and row["rows_test"] == 0:
            refuse("case %r asked for a %r holdout and got no test rows"
                   % (row["case"], row["test_fraction"]))
        if row["test_fraction"] == 0 and row["rows_test"] != 0 \
                and row["workload"] != "cv":
            refuse("case %r asked for no holdout but reports %r test rows"
                   % (row["case"], row["rows_test"]))
    if stage == "none" and not row["function_start_id"]:
        refuse("case %r ran but carries no start fingerprint" % row["case"])
    # weight_id_available, the two ids, the element count and the note must all
    # tell one story.  A row claiming an identity it does not carry -- or
    # carrying one it says is unavailable -- is exactly the confusion the
    # parameter-state identity exists to prevent.
    if row["weight_id_available"]:
        if row["weight_id_note"]:
            refuse("case %r has a weight identity AND an unavailability note: %s"
                   % (row["case"], row["weight_id_note"]))
        if stage == "none":
            if not row["weight_start_id"]:
                refuse("case %r claims a weight identity but carries none"
                       % row["case"])
            if row["weight_elements"] <= 0:
                refuse("case %r claims a weight identity over %r elements"
                       % (row["case"], row["weight_elements"]))
            if row["weight_elements"] != row["params"]:
                refuse("case %r hashed %r weight elements but reports %r "
                       "parameters" % (row["case"], row["weight_elements"],
                                       row["params"]))
    else:
        if row["weight_start_id"] or row["weight_end_id"]:
            refuse("case %r says no weight identity is available yet carries one"
                   % row["case"])
        if not row["weight_id_note"]:
            refuse("case %r has no weight identity and no reason" % row["case"])

    # A completed run must carry BOTH end markers; a throwing one must carry
    # neither, because a model left mid-update is not a state to fingerprint.
    if stage == "none":
        if not row["function_end_id"]:
            refuse("case %r completed but carries no end fingerprint" % row["case"])
        if row["weight_id_available"] and not row["weight_end_id"]:
            refuse("case %r completed but carries no end weight identity"
                   % row["case"])
    elif stage == "training":
        if row["weight_end_id"] or row["function_end_id"]:
            refuse("case %r threw during training but carries an end identity"
                   % row["case"])
    elif stage in ("refused", "setup"):
        if row["elapsed_ns"] != 0:
            refuse("case %r failed at %s but reports elapsed time %r"
                   % (row["case"], stage, row["elapsed_ns"]))
        if row["usable"]:
            refuse("case %r failed at %s but is marked usable" % (row["case"], stage))

    if row["usable"] and row["workload"] == "cv":
        # A WORKFLOW ARM IS USABLE ON DIFFERENT TERMS, and they are stricter,
        # not looser: every fold of every repetition produced predictions --
        # which by cvadapters::trainProcedure's own contract means every fold's
        # fit CONVERGED, never stopped at its ceiling -- and the locked refit
        # produced a scored model.  It carries no iteration or pass count
        # because it ran k*repeats+1 fits and has no single one.
        if row["stop_reason"] != "cv_complete":
            refuse("cv case %r is usable but stopped on %r"
                   % (row["case"], row["stop_reason"]))
        expected_folds = row["cv_folds"] * row["cv_repeats"]
        if row["cv_folds_total"] != expected_folds:
            refuse("cv case %r is usable but ran %r folds, not %r x %r"
                   % (row["case"], row["cv_folds_total"], row["cv_folds"],
                      row["cv_repeats"]))
        if row["cv_folds_ok"] != row["cv_folds_total"]:
            refuse("cv case %r is usable but only %r of %r folds fitted"
                   % (row["case"], row["cv_folds_ok"], row["cv_folds_total"]))
        if row["locked_auc"] is None or row["locked_auc"] < 0:
            refuse("cv case %r is usable but its locked refit produced no score"
                   % row["case"])
        if row["iterations_completed"] is not None \
                or row["full_passes"] is not None:
            refuse("cv case %r reports a single fit's iteration or pass count; "
                   "it ran %r fits" % (row["case"], expected_folds + 1))
        if row["failure_stage"] != "none":
            refuse("case %r is usable but is staged as %r"
                   % (row["case"], row["failure_stage"]))
    elif row["usable"]:
        if not row["finite"]:
            refuse("case %r is usable but not finite" % row["case"])
        if not row["target_reached"]:
            refuse("case %r is usable but did not reach its target" % row["case"])
        if not row["converged"]:
            refuse("case %r is usable but did not converge" % row["case"])
        # A usable fit stopped on the rule it armed. With the matched objective
        # armed that is min_error and nothing else; a plateau-stopped arm is
        # judged by its own rule instead of being held to a name it never used.
        if row["min_stop"]:
            if row["stop_reason"] != "min_error":
                refuse("case %r is usable but stopped on %r, not min_error"
                       % (row["case"], row["stop_reason"]))
        elif row["stop_reason"] != "plateau":
            refuse("case %r armed the plateau rule but stopped on %r"
                   % (row["case"], row["stop_reason"]))
        if row["stop_reason"] in ("max_iterations", "cancelled", "probe_budget"):
            refuse("case %r is usable but ended at %r"
                   % (row["case"], row["stop_reason"]))
        if not isinstance(row["iterations_completed"], int) \
                or row["iterations_completed"] <= 0:
            refuse("case %r is usable but completed %r iterations"
                   % (row["case"], row["iterations_completed"]))
        if not isinstance(row["full_passes"], int) or row["full_passes"] <= 0:
            refuse("case %r is usable but counted %r passes"
                   % (row["case"], row["full_passes"]))
        if row["achieved"] is None or row["achieved"] >= row["target"]:
            refuse("case %r is usable but achieved %r against target %r"
                   % (row["case"], row["achieved"], row["target"]))
        if row["full_passes"] < row["iterations_completed"]:
            refuse("case %r counted fewer passes (%r) than iterations (%r)"
                   % (row["case"], row["full_passes"], row["iterations_completed"]))
        if row["iterations_completed"] > row["ceiling"] + 1:
            refuse("case %r completed %r iterations against a ceiling of %r"
                   % (row["case"], row["iterations_completed"], row["ceiling"]))
        if row["failure_stage"] != "none":
            refuse("case %r is usable but is staged as %r"
                   % (row["case"], row["failure_stage"]))
    else:
        if not row["error"]:
            refuse("case %r is unusable with no reason given" % row["case"])


def validate_group(rows):
    """Every comparison group must be a fair comparison.  Raises Refusal."""
    groups = {}
    for r in rows:
        groups.setdefault(r["comparison_group"], []).append(r)

    for name, members in sorted(groups.items()):
        axes = set(m["group_axis"] for m in members)
        if len(axes) != 1:
            refuse("group %r declares more than one axis: %s"
                   % (name, sorted(axes)))
        axis = axes.pop()
        if axis not in AXIS_FIELDS:
            refuse("group %r declares unknown axis %r" % (name, axis))
        varying = AXIS_FIELDS[axis]

        invariants = [f for f in GROUP_INVARIANTS if f not in varying]
        ref = members[0]
        for m in members[1:]:
            for f in invariants:
                if m[f] != ref[f]:
                    refuse("group %r is not a fair comparison: %s differs "
                           "between %r (%r) and %r (%r), but the group's only "
                           "declared axis is %r"
                           % (name, f, ref["case"], ref[f], m["case"], m[f], axis))

        # Repeated instances of one case must not drift.
        by_case = {}
        for m in members:
            by_case.setdefault(m["case"], []).append(m)
        for case, reps in by_case.items():
            first = reps[0]
            for m in reps[1:]:
                for f in GROUP_INVARIANTS + ["optimizer", "auto_step", "method"]:
                    if m[f] != first[f]:
                        refuse("case %r changed %s between repetitions: %r then %r"
                               % (case, f, first[f], m[f]))

        # A multi-arm group must be comparable at all: its arms must share a
        # PROVEN parameter-state identity.  Where none is available -- BackProp,
        # whose Weights are private -- the group is refused rather than reported
        # as a proven-identical-start comparison.
        if len(by_case) > 1:
            if not all(m["weight_id_available"] for m in members):
                names = sorted(set(m["case"] for m in members
                                   if not m["weight_id_available"]))
                note = next((m["weight_id_note"] for m in members
                             if not m["weight_id_available"]), "")
                refuse("group %r cannot be certified: %s carry no parameter-state "
                       "identity, so identical starts are unproven (%s)"
                       % (name, ", ".join(names), note))
            starts = set(m["weight_start_id"] for m in members)
            if len(starts) != 1:
                refuse("group %r did not start from one parameter state: %s"
                       % (name, sorted(starts)))


# ---------------------------------------------------------------------------
# Statistics -- median/MAD/percentile without third-party libraries


def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return None
    if n % 2:
        return float(s[n // 2])
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def mad(xs):
    """Median absolute deviation: a spread one slow run cannot inflate."""
    if not xs:
        return None
    m = median(xs)
    return median([abs(x - m) for x in xs])


def percentile(xs, p):
    """Nearest-rank: the value reported is a value actually observed."""
    if not xs:
        return None
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(p / 100.0 * (len(s) - 1)))))
    return float(s[k])


# ---------------------------------------------------------------------------
# Probe invocation


def runner_identity():
    """A hash of THIS script, computed at runtime.

    The runner materially controls a campaign: it selects the arms, orders them,
    validates every row and decides what is summarized.  A saved result that
    identifies only the measuring binary cannot answer "what selected and
    filtered these rows?", so the campaign records both.
    """
    import hashlib
    path = os.path.abspath(__file__)
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


def campaign_metadata(identity, seed, cases, plan, timeout):
    """The campaign record written as the first line of an --out file.

    It is distinguished by "record":"campaign"; arm rows carry "schema" and no
    "record", so a reader can tell them apart without positional assumptions.

    `plan` maps each case to (repetitions, reason).  THE REASON IS RECORDED, not
    just the count: a campaign that ran one arm three times and another fifteen
    is a legitimate campaign, and an illegitimate one looks identical from the
    counts alone.  What separates them is whether the choice was made by a
    declared policy from an observation, and that is what this carries.
    """
    return {
        "record": "campaign",
        "schema": SCHEMA_EXPECTED,
        "runner_id": runner_identity(),
        "runner_file": os.path.basename(os.path.abspath(__file__)),
        "probe": dict((f, identity[f]) for f in IDENTITY_FIELDS),
        "orchestration_seed": seed,
        "repetition_policy": [
            {"under_seconds": lim, "repetitions": n, "reason": why}
            for lim, n, why in REPETITION_LADDER
        ],
        "repetitions": dict((c, plan[c][0]) for c in plan),
        "repetition_reasons": dict((c, plan[c][1]) for c in plan),
        "per_arm_timeout_s": timeout,
        "cases": list(cases),
    }


def validate_campaign(meta, identity, expected_runner_id=None):
    """A campaign record must identify the runner AND the binary, and both must
    match what is actually in play.  Raises Refusal."""
    if not isinstance(meta, dict):
        refuse("campaign record is not a JSON object")
    if meta.get("record") != "campaign":
        refuse("campaign record is missing its record marker")
    for f in ("schema", "runner_id", "probe", "orchestration_seed",
              "repetitions", "cases"):
        if f not in meta:
            refuse("campaign record is missing %s" % f)
    if meta["schema"] != SCHEMA_EXPECTED:
        refuse("campaign record has schema %r, this runner expects %d"
               % (meta["schema"], SCHEMA_EXPECTED))
    if not isinstance(meta["runner_id"], str) or not meta["runner_id"]:
        refuse("campaign runner_id is empty or not a string")
    if expected_runner_id is not None and meta["runner_id"] != expected_runner_id:
        refuse("campaign was produced by runner %r but this runner is %r -- a "
               "different orchestrator selected, ordered and filtered those rows"
               % (meta["runner_id"], expected_runner_id))
    if identity is not None:
        for f in IDENTITY_FIELDS:
            if meta["probe"].get(f) != identity[f]:
                refuse("campaign probe %s is %r but the invoked binary reports "
                       "%r" % (f, meta["probe"].get(f), identity[f]))


def find_probe(explicit):
    if explicit:
        if not os.path.isfile(explicit) or not os.access(explicit, os.X_OK):
            sys.exit("refused -- probe: %r is not an executable file" % explicit)
        return os.path.abspath(explicit)
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    for candidate in (os.path.join(root, "build", "optimizer_probe"),
                      os.path.join(root, "build", "Release", "optimizer_probe")):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    sys.exit("refused -- probe: not found; build it or pass --probe PATH")


def probe_identity(probe):
    """Ask the binary what it is.  Never inferred from the working tree: the
    point is to catch a stale binary claiming the current source."""
    out = subprocess.run([probe, "--identity"], capture_output=True, text=True,
                         timeout=60)
    if out.returncode != 0:
        sys.exit("refused -- probe --identity failed: %s" % out.stderr.strip())
    try:
        ident = json.loads(out.stdout.strip())
    except ValueError as exc:
        sys.exit("refused -- probe --identity emitted malformed JSON: %s" % exc)
    for f in ["schema"] + IDENTITY_FIELDS:
        if f not in ident:
            sys.exit("refused -- probe identity is missing %s" % f)
    if ident["schema"] != SCHEMA_EXPECTED:
        sys.exit("refused -- probe schema %r, runner expects %d"
                 % (ident["schema"], SCHEMA_EXPECTED))
    return ident


def list_cases(probe, subset=None):
    """The probe's own case list, optionally narrowed to one half of the table.

    THE PROBE DECIDES MEMBERSHIP, not this script.  Selecting "the Step 0B
    cases" by matching a name prefix here would be a second definition of which
    cases those are, and it would silently drift the first time a case is
    renamed.
    """
    argv = [probe, "--list"] + ([subset] if subset else [])
    out = subprocess.run(argv, capture_output=True, text=True,
                         timeout=60)
    if out.returncode != 0:
        sys.exit("refused -- probe --list failed: %s" % out.stderr.strip())
    names = [l.split()[0] for l in out.stdout.splitlines() if l.strip()]
    if not names:
        sys.exit("refused -- probe --list produced no cases")
    return names


def run_one(probe, case, identity, timeout):
    """Run one arm in its own process.

    One arm per process is deliberate: peak_rss_kb is a PROCESS-CUMULATIVE
    high-water mark, so it only means anything when one arm produced it.

    The revision is NOT passed in.  Whatever the binary was built from is what
    the row must say; overriding it with the working tree's HEAD is how a stale
    binary comes to claim source it does not contain.
    """
    try:
        out = subprocess.run([probe, "--case", case], capture_output=True,
                             text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        # No row is fabricated.  The arm is named, the campaign stops, and the
        # exit status is nonzero: the probe failed its one-row contract.
        sys.exit("refused -- case %r exceeded the %g s per-arm timeout. No row "
                 "was produced; nothing is inferred about its speed. Raise "
                 "--timeout for a genuinely long workload." % (case, timeout))
    if out.returncode != 0:
        sys.exit("refused -- probe failed on case %r (exit %d): %s"
                 % (case, out.returncode, out.stderr.strip()))
    lines = [l for l in out.stdout.splitlines() if l.strip()]
    if len(lines) != 1:
        sys.exit("refused -- probe emitted %d rows for case %r, expected exactly 1"
                 % (len(lines), case))
    try:
        row = json.loads(lines[0])
    except ValueError as exc:
        sys.exit("refused -- probe emitted malformed JSON for %r: %s" % (case, exc))
    try:
        validate_row(row, expected_case=case, identity=identity)
    except Refusal as exc:
        sys.exit("refused -- %s" % exc)
    return row


# ---------------------------------------------------------------------------
# Reporting


def summarize(rows, plan=None):
    """Report the campaign.

    TWO SCOPES, TWO TABLES, NEVER ONE.  An optimizer-only number and a
    whole-workflow number measure different jobs, and printing them in one
    ranked list is how the second gets quoted as the first.  The scope is a
    column as well as a heading, so a copied row still carries it.
    """
    by_scope = {}
    for r in rows:
        by_scope.setdefault(r["timing_scope"], []).append(r)

    for scope in sorted(by_scope):
        scoped = by_scope[scope]
        print()
        if scope == "optimizer":
            print("=== OPTIMIZER-ONLY timing: exactly train(), epilogue "
                  "suppressed ===")
        else:
            print("=== WHOLE-WORKFLOW timing: every fold's fit AND its scoring "
                  "epilogue, plus the locked refit ===")
        print("%-52s %5s %4s %4s %12s %10s %10s %10s %9s" %
              ("case", "scope", "n", "bad", "median ms", "MAD ms", "p10 ms",
               "p90 ms", "passes"))
        print("-" * 132)
        by_case = {}
        for r in scoped:
            by_case.setdefault(r["case"], []).append(r)
        for case in sorted(by_case):
            rs = by_case[case]
            usable = [r for r in rs if r["usable"]]
            bad = len(rs) - len(usable)
            if not usable:
                print("%-52s %5s %4d %4d %12s %10s %10s %10s %9s  <- %s" %
                      (case[:52], scope[:5], len(rs), bad, "-", "-", "-", "-",
                       "-", rs[0]["error"]))
                continue
            ms = [r["elapsed_ns"] / 1e6 for r in usable]
            passes = sorted(set(r["full_passes"] for r in usable))
            print("%-52s %5s %4d %4d %12.3f %10.3f %10.3f %10.3f %9s" %
                  (case[:52], scope[:5], len(rs), bad, median(ms), mad(ms),
                   percentile(ms, 10), percentile(ms, 90),
                   "n/a" if passes == [None] else
                   (passes[0] if len(passes) == 1 else "varies")))

    # --- per group: the fastest RELIABLE arm, and the ratio to canonical -----
    print()
    print("=== comparison groups ===")
    groups = {}
    for r in rows:
        groups.setdefault(r["comparison_group"], []).append(r)

    ambiguous = []
    for name in sorted(groups):
        members = groups[name]
        axis = members[0]["group_axis"]
        start_id = members[0]["weight_start_id"]
        arms = {}
        for m in members:
            arms.setdefault(m["case"], []).append(m)

        stats = {}
        for case, rs in arms.items():
            usable = [r for r in rs if r["usable"]]
            if usable:
                ms = [r["elapsed_ns"] / 1e6 for r in usable]
                stats[case] = (median(ms), percentile(ms, 10),
                               percentile(ms, 90), rs[0]["method"])

        print()
        print("group %s  (axis=%s, endpoint=%s, scope=%s%s)"
              % (name, axis, members[0]["endpoint"], members[0]["timing_scope"],
                 ", start=" + start_id[:12] if start_id else
                 ", no shared start state"))

        # AN ARM THAT CANNOT REACH THE ENDPOINT IS NAMED AND EXCLUDED, never
        # given a ratio.  "It did not finish" is a result about that method,
        # and averaging it in or quietly dropping it both destroy the result.
        failed = sorted(c for c in arms if c not in stats)
        for c in failed:
            print("    %-46s EXCLUDED from ratios: %s"
                  % (arms[c][0]["method"], arms[c][0]["error"]))
        if not stats:
            continue

        ranked = sorted(stats.items(), key=lambda kv: kv[1][0])
        baseline = None
        for case, st in stats.items():
            if st[3] == "canonical":
                baseline = st[0]
        for case, (med, p10, p90, method) in ranked:
            ratio = ("%8.2fx" % (baseline / med)) if baseline else "       -"
            print("    %-46s %10.3f ms   %s vs canonical" % (method, med, ratio))

        # ORDERING AMBIGUITY, reported rather than silently resolved.  With a
        # reduced repetition count the two fastest arms can have overlapping
        # p10/p90 intervals, and then their order is not established by this
        # campaign -- which the reader must be told, not left to infer from a
        # table that looks decisive.
        if len(ranked) >= 2:
            (c1, s1), (c2, s2) = ranked[0], ranked[1]
            if s1[2] >= s2[1]:
                ambiguous.append((name, s1[3], s2[3], s1[0], s2[0]))

    if ambiguous:
        print()
        print("ORDERING NOT ESTABLISHED in %d group(s): the two fastest arms "
              "have overlapping p10-p90 intervals." % len(ambiguous))
        for name, m1, m2, v1, v2 in ambiguous:
            print("    %-46s %s (%.3f ms) vs %s (%.3f ms)"
                  % (name, m1, v1, m2, v2))
        print("    More repetitions would be needed to order these; the "
              "campaign does not claim an order it did not establish.")

    if plan:
        print()
        print("=== repetitions actually run, and why ===")
        for c in sorted(plan):
            print("    %-52s %2d  %s" % (c[:52], plan[c][0], plan[c][1]))


# ---------------------------------------------------------------------------
# Self-test: the validation logic, checked deterministically.


def _good_row(**over):
    r = {
        "schema": 3, "rev": "abc1234", "dirty": True, "source_id": "0123456789abcdef",
        "source_files": 76, "engine_id": "fedcba9876543210", "engine_files": 71,
        "build": "Release/NDEBUG", "case": "c1",
        "comparison_group": "g1",
        "group_axis": "optimizer", "fixture": "linear2", "split": "0xaa",
        "data_id": "0xdd", "data_seed": 0, "test_fraction": 0.0,
        "endpoint": "none", "timing_scope": "optimizer", "workload": "fit",
        "cv_folds": 0, "cv_repeats": 0,
        "model": "simpleprop", "arch": "3", "loss": "xentropy", "rows": 240,
        "rows_total": 240, "rows_test": 0,
        "inputs": 2, "params": 13, "weight_seed": 7, "weight_id_available": True,
        "weight_start_id": "0xw1", "weight_end_id": "0xw2", "weight_elements": 13,
        "weight_id_note": "", "function_start_id": "0xf1", "function_end_id": "0xf2",
        "optimizer": 0, "lbfgs_memory": 5, "method": "canonical",
        "optimizer_name": "canonical", "mode": "batch", "eta": 0.5,
        "auto_step": False, "decay_on": False, "decay": 0.0, "grad_stop": True,
        "auto_stop": False, "min_stop": True,
        "target": 0.35, "achieved": 0.30, "ceiling": 4000, "iteration_index": 118,
        "iterations_completed": 119, "full_passes": 119, "elapsed_ns": 2300000,
        "peak_rss_kb": 2000, "stop_reason": "min_error",
        "heldout_error": -1, "cv_auc": -1, "locked_auc": -1,
        "cv_folds_ok": 0, "cv_folds_total": 0, "converged": True,
        "target_reached": True, "finite": True, "usable": True,
        "failure_stage": "none", "error": "",
    }
    r.update(over)
    return r


def self_test():
    fails = []

    def ok(what, fn, must_refuse=True):
        try:
            fn()
            refused = False
            err = ""
        except Refusal as exc:
            refused = True
            err = str(exc)
        if refused == must_refuse:
            print("ok - %s" % what)
        else:
            print("FAIL - %s (refused=%s, expected refusal=%s) %s"
                  % (what, refused, must_refuse, err))
            fails.append(what)

    # The control first: a valid row must PASS, or every refusal below is
    # meaningless because everything refuses.
    ok("a valid row is accepted", lambda: validate_row(_good_row()),
       must_refuse=False)
    ok("a valid single group is accepted",
       lambda: validate_group([_good_row(), _good_row(case="c2", optimizer=2,
                                                      optimizer_name="shanno")]),
       must_refuse=False)

    ok("a missing field is refused",
       lambda: validate_row({k: v for k, v in _good_row().items() if k != "usable"}))
    ok("a wrong schema is refused", lambda: validate_row(_good_row(schema=1)))
    ok("a non-boolean boolean is refused",
       lambda: validate_row(_good_row(usable="yes")))
    ok("a non-integer integer is refused",
       lambda: validate_row(_good_row(rows="240")))
    ok("a negative elapsed time is refused",
       lambda: validate_row(_good_row(elapsed_ns=-1)))
    ok("a row for the wrong case is refused",
       lambda: validate_row(_good_row(), expected_case="other"))
    ok("an optimizer/name mismatch is refused",
       lambda: validate_row(_good_row(optimizer=1)))
    # THE FOUR IDENTITY QUESTIONS, each with its own synthetic mismatch. No real
    # repository file is touched to demonstrate any of them.
    def ident(**over):
        base = {"rev": "abc1234", "dirty": True, "source_id": "0123456789abcdef",
                "source_files": 76, "engine_id": "fedcba9876543210",
                "engine_files": 71, "build": "Release/NDEBUG"}
        base.update(over)
        return base

    ok("a matching binary identity is accepted (the control)",
       lambda: validate_row(_good_row(), identity=ident()), must_refuse=False)
    ok("an ENGINE source change is refused (source_id moves)",
       lambda: validate_row(_good_row(), identity=ident(source_id="ENGINEEDIT")))
    ok("a changed source SET is refused (source_files moves)",
       lambda: validate_row(_good_row(), identity=ident(source_files=75)))
    # THE ENGINE IDENTITY IS ITS OWN QUESTION. source_id moves whenever the
    # committed target table is rewritten, which is not a change to how the
    # engine trains; engine_id moves only when src/ does. A campaign judged
    # against a table characterized on a DIFFERENT engine is not matched.
    ok("an engine change is refused (engine_id moves)",
       lambda: validate_row(_good_row(), identity=ident(engine_id="0" * 16)))
    ok("a changed engine source SET is refused (engine_files moves)",
       lambda: validate_row(_good_row(), identity=ident(engine_files=70)))
    ok("a different revision is refused",
       lambda: validate_row(_good_row(), identity=ident(rev="deadbee")))
    ok("a different dirty state is refused",
       lambda: validate_row(_good_row(), identity=ident(dirty=False)))
    ok("a different build type is refused",
       lambda: validate_row(_good_row(), identity=ident(build="Debug/asserts")))

    # The RUNNER identity: a campaign record must name the orchestrator that
    # produced it, and a different one is refused.
    meta = {"record": "campaign", "schema": 3, "runner_id": "aaaabbbbccccdddd",
            "probe": ident(), "orchestration_seed": 42, "repetitions": 3,
            "cases": ["c1"]}
    ok("a matching campaign record is accepted (the control)",
       lambda: validate_campaign(meta, ident(), "aaaabbbbccccdddd"),
       must_refuse=False)
    ok("a DIFFERENT RUNNER is refused",
       lambda: validate_campaign(meta, ident(), "9999999999999999"))
    ok("a campaign whose probe identity differs is refused",
       lambda: validate_campaign(meta, ident(source_id="OTHER"),
                                 "aaaabbbbccccdddd"))
    ok("a campaign with no record marker is refused",
       lambda: validate_campaign({"schema": 3, "runner_id": "x", "probe": ident(),
                                  "orchestration_seed": 1, "repetitions": 1,
                                  "cases": []}, ident()))
    ok("a campaign missing its runner id is refused",
       lambda: validate_campaign({"record": "campaign", "schema": 3,
                                  "probe": ident(), "orchestration_seed": 1,
                                  "repetitions": 1, "cases": []}, ident()))
    ok("a campaign with an empty runner id is refused",
       lambda: validate_campaign({"record": "campaign", "schema": 3,
                                  "runner_id": "", "probe": ident(),
                                  "orchestration_seed": 1, "repetitions": 1,
                                  "cases": []}, ident()))

    # --- numeric, nullable and domain families, each accepted AND rejected ---
    ok("a numeric field as a malformed string is refused",
       lambda: validate_row(_good_row(eta="0.5")))
    ok("a BOOLEAN masquerading as a number is refused",
       lambda: validate_row(_good_row(eta=True)))
    ok("a BOOLEAN masquerading as an integer is refused",
       lambda: validate_row(_good_row(params=True)))
    ok("a non-finite numeric field is refused",
       lambda: validate_row(_good_row(decay=float("inf"))))
    ok("a NaN numeric field is refused",
       lambda: validate_row(_good_row(decay=float("nan"))))
    ok("eta outside [0,1] is refused", lambda: validate_row(_good_row(eta=1.5)))
    ok("a negative decay is refused", lambda: validate_row(_good_row(decay=-0.1)))
    ok("a target of exactly 0 is refused",
       lambda: validate_row(_good_row(target=0.0)))
    ok("a target of exactly 1 is refused",
       lambda: validate_row(_good_row(target=1.0, achieved=0.5)))
    ok("a negative count is refused", lambda: validate_row(_good_row(rows=-1)))
    ok("a negative elapsed time is refused",
       lambda: validate_row(_good_row(elapsed_ns=-5)))
    ok("a zero ceiling is refused", lambda: validate_row(_good_row(ceiling=0)))
    ok("an out-of-range optimizer number is refused",
       lambda: validate_row(_good_row(optimizer=9, optimizer_name="canonical")))
    ok("an unknown model is refused", lambda: validate_row(_good_row(model="mlp")))
    ok("an unknown mode is refused", lambda: validate_row(_good_row(mode="hybrid")))
    ok("an unknown loss is refused", lambda: validate_row(_good_row(loss="hinge")))
    ok("an unknown stop reason is refused",
       lambda: validate_row(_good_row(stop_reason="vibes", usable=False,
                                      error="x")))
    ok("an empty case name is refused", lambda: validate_row(_good_row(case="")))
    ok("an empty comparison group is refused",
       lambda: validate_row(_good_row(comparison_group="")))

    # Nullable fields: null is legal, a bool is not, a negative is not.
    ok("a null peak_rss_kb is accepted (unavailable on this platform)",
       lambda: validate_row(_good_row(peak_rss_kb=None)), must_refuse=False)
    ok("a boolean in a nullable count is refused",
       lambda: validate_row(_good_row(peak_rss_kb=True)))
    ok("a negative nullable count is refused",
       lambda: validate_row(_good_row(peak_rss_kb=-3)))
    ok("a null full_passes on an unusable row is accepted",
       lambda: validate_row(_good_row(usable=False, error="refused -- model",
                                      failure_stage="refused", elapsed_ns=0,
                                      full_passes=None,
                                      iterations_completed=None,
                                      weight_start_id="", weight_end_id="",
                                      weight_id_available=False,
                                      weight_id_note="unavailable",
                                      function_start_id="", function_end_id="",
                                      split="", stop_reason="none",
                                      converged=False, target_reached=False,
                                      finite=False, achieved=None)),
       must_refuse=False)
    ok("a null achieved with finite=true is refused",
       lambda: validate_row(_good_row(achieved=None)))
    ok("a finite achieved with finite=false is refused",
       lambda: validate_row(_good_row(finite=False, usable=False, error="x",
                                      achieved=0.3)))

    # Weight-identity consistency, both directions.
    ok("claiming an identity while carrying a note is refused",
       lambda: validate_row(_good_row(weight_id_note="unavailable")))
    ok("denying an identity while carrying one is refused",
       lambda: validate_row(_good_row(weight_id_available=False,
                                      weight_id_note="none")))
    ok("no identity and no reason is refused",
       lambda: validate_row(_good_row(weight_id_available=False,
                                      weight_start_id="", weight_end_id="",
                                      weight_id_note="")))
    ok("a weight element count disagreeing with params is refused",
       lambda: validate_row(_good_row(weight_elements=12)))
    ok("a completed row with no end fingerprint is refused",
       lambda: validate_row(_good_row(function_end_id="")))
    ok("a completed row with no end weight identity is refused",
       lambda: validate_row(_good_row(weight_end_id="")))
    ok("a training fault carrying an end identity is refused",
       lambda: validate_row(_good_row(failure_stage="training", usable=False,
                                      error="exception during training")))
    ok("a refused row reporting elapsed time is refused",
       lambda: validate_row(_good_row(failure_stage="refused", usable=False,
                                      error="refused -- model", elapsed_ns=17)))
    ok("a training fault WITHOUT an end identity is accepted",
       lambda: validate_row(_good_row(failure_stage="training", usable=False,
                                      error="exception during training",
                                      weight_end_id="", function_end_id="",
                                      converged=False, target_reached=False,
                                      finite=False, achieved=None)),
       must_refuse=False)
    ok("an unknown failure stage is refused",
       lambda: validate_row(_good_row(failure_stage="weird")))
    ok("an executed row with no split identity is refused",
       lambda: validate_row(_good_row(split="")))
    ok("a run with no start fingerprint is refused",
       lambda: validate_row(_good_row(function_start_id="")))
    ok("a claimed-but-absent weight identity is refused",
       lambda: validate_row(_good_row(weight_start_id="")))

    # The usable predicate, one violated clause at a time.
    ok("usable but not finite is refused",
       lambda: validate_row(_good_row(finite=False)))
    ok("usable but target not reached is refused",
       lambda: validate_row(_good_row(target_reached=False)))
    ok("usable but not converged is refused",
       lambda: validate_row(_good_row(converged=False)))
    ok("usable but ceiling-stopped is refused",
       lambda: validate_row(_good_row(stop_reason="max_iterations")))
    ok("usable but cancelled is refused",
       lambda: validate_row(_good_row(stop_reason="cancelled")))
    ok("usable with zero completed iterations is refused",
       lambda: validate_row(_good_row(iterations_completed=0)))
    ok("usable with no passes is refused",
       lambda: validate_row(_good_row(full_passes=0)))
    ok("usable but achieved >= target is refused",
       lambda: validate_row(_good_row(achieved=0.40)))
    ok("fewer passes than iterations is refused",
       lambda: validate_row(_good_row(full_passes=5)))
    ok("more iterations than the ceiling allows is refused",
       lambda: validate_row(_good_row(ceiling=10)))
    ok("usable but staged as a failure is refused",
       lambda: validate_row(_good_row(failure_stage="training")))
    ok("unusable with no reason is refused",
       lambda: validate_row(_good_row(usable=False, error="")))
    ok("unusable WITH a reason is accepted",
       lambda: validate_row(_good_row(usable=False, error="ceiling exhausted",
                                      stop_reason="max_iterations",
                                      converged=False, target_reached=False)),
       must_refuse=False)

    # Comparison groups.
    ok("a group varying eta as well as optimizer is refused",
       lambda: validate_group([_good_row(),
                               _good_row(case="c2", optimizer=2,
                                         optimizer_name="shanno", eta=0.1)]))
    ok("a group varying the grad_stop branch under an optimizer axis is refused",
       lambda: validate_group([_good_row(),
                               _good_row(case="c2", optimizer=2,
                                         optimizer_name="shanno", grad_stop=False)]))
    ok("a group starting from two parameter states is refused",
       lambda: validate_group([_good_row(),
                               _good_row(case="c2", optimizer=2,
                                         optimizer_name="shanno",
                                         weight_start_id="0xOTHER")]))
    ok("a multi-arm group with no parameter-state identity is refused",
       lambda: validate_group([
           _good_row(weight_id_available=False, weight_start_id="",
                     weight_id_note="a model with no accessor"),
           _good_row(case="c2", optimizer=2, optimizer_name="shanno",
                     weight_id_available=False, weight_start_id="",
                     weight_id_note="a model with no accessor")]))
    ok("a group declaring two axes is refused",
       lambda: validate_group([_good_row(),
                               _good_row(case="c2", group_axis="auto_step")]))
    ok("an unknown axis is refused",
       lambda: validate_group([_good_row(group_axis="vibes")]))
    ok("a case drifting between repetitions is refused",
       lambda: validate_group([_good_row(), _good_row(target=0.20)]))
    ok("an auto_step group varying auto_step alone is accepted",
       lambda: validate_group([
           _good_row(group_axis="auto_step", comparison_group="g2"),
           _good_row(group_axis="auto_step", comparison_group="g2", case="c2",
                     auto_step=True)]),
       must_refuse=False)
    ok("a single-arm group with no weight identity is accepted (nothing compared)",
       lambda: validate_group([_good_row(weight_id_available=False,
                                         weight_start_id="")]),
       must_refuse=False)

    # --- Step 0B mechanics -------------------------------------------------
    #
    # A cv row is a legitimate, complete row of its own shape.  This one is the
    # control every refusal below is measured against: if it were itself
    # refused, the refusals would prove nothing.
    def _cv_row(**over):
        base = dict(workload="cv", timing_scope="workflow",
                    cv_folds=5, cv_repeats=2, cv_folds_ok=10, cv_folds_total=10,
                    stop_reason="cv_complete", iterations_completed=None,
                    full_passes=None, cv_auc=0.71, locked_auc=0.70,
                    achieved=0.71, target=0.32, test_fraction=0.25,
                    rows=4500, rows_total=6000, rows_test=1500,
                    weight_id_available=False, weight_start_id="",
                    weight_end_id="",
                    weight_id_note="a cv arm has no single starting state",
                    weight_elements=0, endpoint="none", min_stop=False,
                    auto_stop=True)
        base.update(over)
        return _good_row(**base)

    ok("a complete cv row is accepted (the control)",
       lambda: validate_row(_cv_row()), must_refuse=False)

    # THE MISLABELLING THIS WHOLE FIELD EXISTS TO PREVENT.  A cv clock covers
    # each fold's scoring epilogue; calling that an optimizer timing would let
    # a workflow number be quoted as a per-fit one.
    ok("a cv arm claiming optimizer scope is refused",
       lambda: validate_row(_cv_row(timing_scope="optimizer")))
    ok("a fit arm reporting cv folds is refused",
       lambda: validate_row(_good_row(cv_folds=5)))
    ok("a fit arm counting fold outcomes is refused",
       lambda: validate_row(_good_row(cv_folds_total=5)))
    ok("a usable cv arm whose folds did not all fit is refused",
       lambda: validate_row(_cv_row(cv_folds_ok=9)))
    ok("a usable cv arm running the wrong number of folds is refused",
       lambda: validate_row(_cv_row(cv_folds_ok=9, cv_folds_total=9)))
    ok("a usable cv arm with no locked-refit score is refused",
       lambda: validate_row(_cv_row(locked_auc=-1)))
    ok("a cv arm reporting a single fit's pass count is refused",
       lambda: validate_row(_cv_row(full_passes=99)))
    ok("an unknown endpoint is refused",
       lambda: validate_row(_good_row(endpoint="whenever")))
    ok("an unknown timing scope is refused",
       lambda: validate_row(_good_row(timing_scope="wall")))
    ok("an unknown workload is refused",
       lambda: validate_row(_good_row(workload="stepwise")))

    # THE METHOD NAME IS DERIVED and cannot disagree with what it names --
    # "canonical" and "canonical-autostep" are two methods, not one.
    ok("a method name disagreeing with auto_step is refused",
       lambda: validate_row(_good_row(auto_step=True)))
    ok("a method name disagreeing with the optimizer is refused",
       lambda: validate_row(_good_row(optimizer=2, optimizer_name="shanno")))
    ok("an autostep row naming its method correctly is accepted",
       lambda: validate_row(_good_row(auto_step=True,
                                      method="canonical-autostep")),
       must_refuse=False)

    # THE SPLIT MUST ADD UP.
    ok("a split training on more rows than the dataset holds is refused",
       lambda: validate_row(_good_row(rows=5000, rows_test=1500,
                                      rows_total=6000, test_fraction=0.25)))
    ok("a requested holdout that produced no test rows is refused",
       lambda: validate_row(_good_row(test_fraction=0.25, rows_test=0)))
    ok("test rows with no requested holdout are refused",
       lambda: validate_row(_good_row(test_fraction=0.0, rows_test=10)))
    ok("an executed row with no data identity is refused",
       lambda: validate_row(_good_row(data_id="")))
    ok("a real holdout that adds up is accepted",
       lambda: validate_row(_good_row(test_fraction=0.25, rows=4500,
                                      rows_test=1500, rows_total=6000,
                                      data_seed=20260804)),
       must_refuse=False)

    # ENDPOINTS AND SCOPES CANNOT DRIFT WITHIN A GROUP.  A practical arm and a
    # strict arm are racing to different objectives; an optimizer-only arm and
    # a workflow arm are doing different jobs.  Either mixture would produce a
    # ratio between two things that were never the same race.
    ok("a group mixing the practical and strict endpoints is refused",
       lambda: validate_group([
           _good_row(endpoint="practical"),
           _good_row(endpoint="strict", case="c2", optimizer=2,
                     method="shanno", optimizer_name="shanno")]))
    ok("a group mixing optimizer-only and workflow scope is refused",
       lambda: validate_group([
           _good_row(timing_scope="optimizer"),
           _good_row(timing_scope="workflow", case="c2", optimizer=2,
                     method="shanno", optimizer_name="shanno")]))
    ok("a group mixing fit and cv workloads is refused",
       lambda: validate_group([_good_row(), _cv_row(case="c2")]))
    ok("a group mixing repetition counts is refused",
       lambda: validate_group([
           _cv_row(comparison_group="gcv"),
           _cv_row(comparison_group="gcv", case="c2", cv_repeats=3,
                   cv_folds_ok=15, cv_folds_total=15, optimizer=2,
                   method="shanno", optimizer_name="shanno")]))
    ok("a group differing only in data_seed is refused",
       lambda: validate_group([
           _good_row(),
           _good_row(case="c2", data_seed=99, optimizer=2, method="shanno",
                     optimizer_name="shanno")]))

    # THE `method` AXIS frees the optimizer AND the step-size search, and
    # nothing else -- that is what makes the brief's four-method comparison a
    # legitimate group rather than a relaxation of the invariant.
    ok("a method group varying optimizer and auto_step together is accepted",
       lambda: validate_group([
           _good_row(group_axis="method", comparison_group="gm"),
           _good_row(group_axis="method", comparison_group="gm", case="c2",
                     auto_step=True, method="canonical-autostep"),
           _good_row(group_axis="method", comparison_group="gm", case="c3",
                     optimizer=2, optimizer_name="shanno", method="shanno")]),
       must_refuse=False)
    ok("a method group varying eta as well is still refused",
       lambda: validate_group([
           _good_row(group_axis="method", comparison_group="gm"),
           _good_row(group_axis="method", comparison_group="gm", case="c2",
                     optimizer=2, optimizer_name="shanno", method="shanno",
                     eta=0.1)]))

    # A WORKFLOW ARM IS NOT A MATCHED-ENDPOINT RACE, and must not be able to
    # dress as one.  These are the pair that keep the cv arms honest about what
    # they measured.
    ok("a cv arm claiming the practical endpoint is refused",
       lambda: validate_row(_cv_row(endpoint="practical")))
    ok("an arm with no stopping rule at all is refused",
       lambda: validate_row(_cv_row(auto_stop=False)))
    # THE SABOTAGE THAT FOUND THIS: dropping the reasons made the check crash
    # rather than fail, so it printed nothing at all. .get() keeps a missing
    # field reportable.
    ok("a plateau-stopped arm reported as min_error-stopped is refused",
       lambda: validate_row(_good_row(min_stop=False, auto_stop=True,
                                      endpoint="none")))
    ok("a fit arm stopping on a plateau it armed is accepted",
       lambda: validate_row(_good_row(min_stop=False, auto_stop=True,
                                      endpoint="none", stop_reason="plateau")),
       must_refuse=False)
    ok("a group mixing armed stopping rules is refused",
       lambda: validate_group([
           _good_row(),
           _good_row(case="c2", min_stop=False, auto_stop=True,
                     endpoint="none", stop_reason="plateau",
                     optimizer=2, method="shanno", optimizer_name="shanno")]))

    # UNUSABLE RUNS CANNOT ENTER A SPEED SUMMARY.  Checked on the summary code
    # itself rather than on a comment about it: a group whose fastest-looking
    # arm never reached its endpoint must not be given a ratio.
    import io
    import contextlib
    buf = io.StringIO()
    fast_but_failed = _good_row(
        case="cfail", optimizer=2, optimizer_name="shanno", method="shanno",
        usable=False, converged=False, target_reached=False,
        stop_reason="max_iterations", achieved=0.9, elapsed_ns=1,
        error="ceiling exhausted without reaching target")
    with contextlib.redirect_stdout(buf):
        summarize([_good_row(), _good_row(), fast_but_failed, fast_but_failed])
    text = buf.getvalue()
    if "EXCLUDED from ratios" in text and "0.001 ms" not in text:
        print("ok - an arm that never reached its endpoint is excluded from "
              "the ratios and named")
    else:
        print("FAIL - a failed arm reached the speed summary")
        fails.append("summary excludes unusable")

    # THE REPETITION POLICY IS REPRESENTED HONESTLY: the count is chosen by a
    # declared ladder from an observation, and the reason carries the
    # observation with it.  A count with no reason is the shape that lets three
    # repetitions look like a decision instead of a measurement.
    ladder_ok = (reps_for(0.4)[0] == 15 and reps_for(12.0)[0] == 5
                 and reps_for(600.0)[0] == 3
                 and "0.400" in reps_for(0.4)[1]
                 and "600.000" in reps_for(600.0)[1]
                 # An arm that never reached its endpoint is recorded once at
                 # every duration, because repeating it measures nothing.
                 and reps_for(0.4, False)[0] == 1
                 and reps_for(600.0, False)[0] == 1
                 and "did not reach its endpoint" in reps_for(0.4, False)[1])
    if ladder_ok:
        print("ok - the repetition ladder picks 15/5/3 by measured warm-up cost "
              "and records the observation")
    else:
        print("FAIL - the repetition ladder is wrong")
        fails.append("repetition ladder")

    meta = campaign_metadata(ident(), 7, ["c1", "c2"],
                             {"c1": (15, "short cell: warm-up under 1 s "
                                         "(warm-up 0.100 s)"),
                              "c2": (3, "expensive cell (warm-up 400.000 s)")},
                             900.0)
    # .get(), not [], deliberately: a missing reason is the exact defect this
    # checks for, and indexing would raise out of the whole self-test instead
    # of reporting one failure. Found by sabotage -- dropping the reasons
    # produced no FAIL line at all, because the check crashed before printing.
    meta_ok = (meta["repetitions"] == {"c1": 15, "c2": 3}
               and "warm-up 400.000 s" in meta.get("repetition_reasons", {})
                                              .get("c2", "")
               and len(meta.get("repetition_policy", [])) == len(REPETITION_LADDER))
    if meta_ok:
        print("ok - the campaign record carries per-case repetition counts, "
              "their reasons, and the policy itself")
    else:
        print("FAIL - the campaign record hides how repetitions were chosen")
        fails.append("campaign repetition record")

    # The statistics, on values whose answers are known by hand.
    stats_ok = (median([1, 2, 3]) == 2.0 and median([1, 2, 3, 4]) == 2.5
                and mad([1, 2, 3, 100]) == 1.0
                and percentile([1, 2, 3, 4, 5], 10) == 1.0
                and percentile([1, 2, 3, 4, 5], 90) == 5.0
                and median([]) is None and mad([]) is None)
    if stats_ok:
        print("ok - median, MAD and nearest-rank percentiles are correct")
    else:
        print("FAIL - the statistics are wrong")
        fails.append("statistics")

    print()
    print("FAILURES: %d" % len(fails) if fails else "all passed (0 failures)")
    return 1 if fails else 0


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", help="path to the optimizer_probe binary")
    ap.add_argument("--reps", type=int, default=None,
                    help="force this many repetitions for EVERY case, overriding "
                         "the cost-scaled policy. Omit to let each case's own "
                         "warm-up duration choose (15 / 5 / 3)")
    ap.add_argument("--seed", type=int, default=None,
                    help="orchestration seed; one is chosen and REPORTED if omitted")
    ap.add_argument("--case", action="append", default=[],
                    help="restrict to this case (repeatable)")
    ap.add_argument("--out", help="write raw JSONL rows here")
    ap.add_argument("--pilot", action="store_true",
                    help="the Step 0A MECHANICS pilot only, at 3 repetitions: "
                         "for validating the plumbing, not for evidence")
    ap.add_argument("--step0b", action="store_true",
                    help="restrict to the Step 0B workload matrix (the Civic "
                         "Choice application benchmark, the scaling series and "
                         "the repeated-fit consumer)")
    ap.add_argument("--screen", action="store_true",
                    help="restrict to the Phase 3 candidate screen: the L-BFGS "
                         "prototype against the Shanno incumbent, on Civic "
                         "Choice 6k/h4 at the committed practical endpoint")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="per-arm timeout in seconds (default 120; raise it "
                         "deliberately for a genuinely long Step 0B workload)")
    ap.add_argument("--self-test", action="store_true",
                    help="check this script's own validation logic and exit")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    if args.pilot:
        args.reps = 3
    if args.reps is not None and args.reps < 1:
        sys.exit("refused -- reps: must be at least 1")
    if args.timeout <= 0:
        sys.exit("refused -- timeout: must be positive")

    probe = find_probe(args.probe)
    identity = probe_identity(probe)

    subset = ("--step0b" if args.step0b
              else "--screen" if args.screen
              else "--pilot" if args.pilot
              else None)
    available = list_cases(probe, subset)
    if args.case:
        unknown = [c for c in args.case if c not in available]
        if unknown:
            sys.exit("refused -- case: unknown case(s) %s; available: %s"
                     % (", ".join(unknown), ", ".join(available)))
        cases = list(args.case)
    else:
        cases = available

    seed = args.seed if args.seed is not None else random.randrange(1, 2 ** 31)
    rng = random.Random(seed)

    print("probe        : %s" % probe)
    print("binary       : rev=%s dirty=%s source_id=%s (%d files) build=%s"
          % (identity["rev"], identity["dirty"], identity["source_id"],
             identity["source_files"], identity["build"]))
    print("runner       : %s (%s)"
          % (runner_identity(), os.path.basename(os.path.abspath(__file__))))
    if identity["dirty"]:
        print("               (dirty tree: rev names a commit this binary was "
              "NOT built from; source_id is the authority)")
    print("cases        : %d" % len(cases))
    print("orchestration seed : %d   (pass --seed %d to reproduce this order)"
          % (seed, seed))

    # WARM-UP, discarded AS A MEASUREMENT and kept AS AN OBSERVATION. The first
    #    run of any arm pays for page faults and a cold instruction cache, and
    #    charging that to whichever case happens to run first is the easiest way
    #    to manufacture a false difference. But how long it took is exactly the
    #    fact the repetition policy needs, and it has already been paid for.
    print("warm-up      : running each case once, discarded")
    plan = {}
    for c in cases:
        t0 = time.monotonic()
        # The warm-up row is DISCARDED as a measurement and READ as an
        #    observation: how long the arm takes, and whether it finishes at
        #    all. Both already paid for.
        warm = run_one(probe, c, identity, args.timeout)
        elapsed = time.monotonic() - t0
        if args.reps is not None:
            plan[c] = (args.reps, "forced by --reps %d (warm-up %.3f s)"
                       % (args.reps, elapsed))
        else:
            plan[c] = reps_for(elapsed, warm["usable"])
        sys.stdout.write("\rwarm-up      : %-52s %6.1f s %s"
                         % (c[:52], elapsed, "ok" if warm["usable"] else "FAILED"))
        sys.stdout.flush()
    print()

    print("repetitions  : chosen per case from the warm-up's own duration")
    for c in sorted(plan):
        print("    %-52s %2d  %s" % (c[:52], plan[c][0], plan[c][1]))
    print("per-arm timeout: %g s" % args.timeout)

    # Interleaved to the DEEPEST plan: repetition r runs every case that still
    #    owes a run, in an independently shuffled order. A cheap arm therefore
    #    keeps being sampled alongside an expensive one instead of all of its
    #    runs landing in one contiguous stretch of machine weather.
    rows = []
    max_reps = max(n for n, _ in plan.values())
    for rep in range(max_reps):
        order = [c for c in cases if plan[c][0] > rep]
        rng.shuffle(order)
        for c in order:
            rows.append(run_one(probe, c, identity, args.timeout))
        sys.stdout.write("\rmeasured     : round %d/%d (%d arms)"
                         % (rep + 1, max_reps, len(order)))
        sys.stdout.flush()
    print()

    meta = campaign_metadata(identity, seed, cases, plan, args.timeout)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(json.dumps(meta) + "\n")   # line 1: who ran this
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print("raw rows     : %s (1 campaign record + %d arm rows)"
              % (args.out, len(rows)))
    else:
        print("raw rows     : not retained (pass --out FILE to keep them)")

    # COMPARISON FAIRNESS IS ENFORCED, NOT WARNED ABOUT. A group whose arms do
    #    not describe the same work, or cannot be shown to have started from the
    #    same parameter state, cannot support a timing conclusion -- so the
    #    result set is refused rather than summarized with a caveat nobody reads.
    #    Groups are checked only among the cases actually selected.
    try:
        validate_group(rows)
    except Refusal as exc:
        print()
        print("REFUSED -- %s" % exc)
        print("No timing summary is produced: these rows cannot support a "
              "comparison.")
        return 1

    summarize(rows, plan)

    unusable = [r for r in rows if not r["usable"]]
    if unusable:
        print()
        print("%d unusable arm(s) -- reported, never averaged:" % len(unusable))
        seen = set()
        for r in unusable:
            key = (r["case"], r["failure_stage"], r["stop_reason"], r["error"])
            if key not in seen:
                seen.add(key)
                print("  %-34s %-9s %-15s %s"
                      % (r["case"], r["failure_stage"], r["stop_reason"], r["error"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
