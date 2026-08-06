cmake_minimum_required(VERSION 3.16)
project(neuron VERSION 3.0.0 LANGUAGES CXX)

# Default to Release: a bare `cmake -B build` otherwise builds with NO
# optimization at all, and the engine runs ~15x slower -- discovered 2026-07-21
# when a locally reconfigured build dir had an empty CMAKE_BUILD_TYPE and every
# timing measured against it was wrong. CI builds Release explicitly.
#
# Asserts vanish under -DNDEBUG, so nothing the engine must not do may be
# guarded by one: handlers validate, Matrix bounds checks are unconditional
# throws, and since 2026-08-01 so are the bounds and dimension checks in
# vector_ops (D5). This comment used to name Matrix alone, which read as
# though it were the whole guarantee -- every vector_ops precondition was an
# assert, and because this project builds Release by default, not one of them
# had ever executed in the gate chain. Single-config generators only -- MSVC
# picks its config at build time (--config Release, as CI does).
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(GSL REQUIRED)
find_package(Threads REQUIRED)

# Embed the GUI page (src/gui_page.html) into the binary as a raw string
file(READ src/gui_page.html GUI_PAGE_CONTENT)
configure_file(src/gui_page.h.in generated/gui_page.h @ONLY)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS src/gui_page.html)

# --- The engine, as LAYERED static libraries -------------------------------
#
# Each source is compiled ONCE and linked wherever it is needed. Before this,
# every test target listed its own copy of the engine sources: 208 compilation
# units for 31 files, which is what stalled the hosted macOS CI runner when they
# all built at once (docs/refactor_audit.md section 10).
#
# The layering is not decoration, and it is not one library on purpose. ONLY
# network.cpp includes GSL. Several tests are documented as proving that the
# statistics and data layers do not depend on it -- "Links only the statistics
# core, no GSL needed" -- and a monolithic library would erase that proof by
# handing GSL to everything. Here the dependency graph ENFORCES it: a target
# that links neuron_core cannot acquire GSL, because nothing below
# neuron_engine has it to give.
#
# Dependencies point downward, as everywhere else in this project (rule 6):
#
#     neuron_core     numerical primitives + the statistics container
#     neuron_data     splitting and dataset materialization
#     neuron_infer    ROC covariance and evaluation-design types
#     neuron_engine   models and training            <-- GSL enters here
#     neuron_eval     cross-validation and reporting
#     neuron          the driver and the embedded GUI

add_library(neuron_core STATIC
    src/utility.cpp src/vector_ops.cpp src/matrix.cpp src/stats.cpp
    src/twoset.cpp)
target_include_directories(neuron_core PUBLIC src)

add_library(neuron_data STATIC src/split.cpp src/dataset.cpp)
target_link_libraries(neuron_data PUBLIC neuron_core)

add_library(neuron_infer STATIC
    src/auccov.cpp src/delong.cpp src/clustered_auc.cpp src/evaldesign.cpp)
target_link_libraries(neuron_infer PUBLIC neuron_core)

# The first layer that needs GSL -- network.cpp's eigenvalue routines -- and so
# the first that passes it on. PUBLIC because a caller linking the engine links
# GSL transitively; that is the boundary, stated once.
add_library(neuron_engine STATIC
    src/model.cpp src/iterative.cpp src/network.cpp
    src/onehidden.cpp src/simpleprop.cpp src/bareprop.cpp
    src/backprop.cpp src/logistic.cpp
    src/dfa.cpp src/ldfa.cpp src/qdfa.cpp
    src/netclone.cpp src/modelfactory.cpp src/regressnet.cpp
    src/autoalgo.cpp src/obd.cpp)
target_link_libraries(neuron_engine PUBLIC neuron_data GSL::gsl GSL::gslcblas)

add_library(neuron_eval STATIC
    src/crossval.cpp src/cvadapters.cpp src/cvreport.cpp)
target_link_libraries(neuron_eval PUBLIC neuron_engine neuron_infer)

# The two process boundaries: the asynchronous-job lifecycle the GUI's four
# long operations share, and the top-level exception guard the CLI enters
# through. Neither knows anything about the engine -- this library depends on
# the standard library and threads alone, which is what makes both of them
# linkable by a test without dragging GSL in. They are separate source and
# header pairs on purpose (see procguard.h).
add_library(neuron_job STATIC src/asyncjob.cpp src/procguard.cpp)
target_include_directories(neuron_job PUBLIC src)
target_link_libraries(neuron_job PUBLIC Threads::Threads)

# The driver and the embedded GUI. gui.cpp is the only translation unit that
# needs the generated page header and the vendored httplib, so those include
# directories stay PRIVATE to this target rather than leaking into a library.
add_executable(neuron src/neuron.cpp src/gui.cpp)
target_include_directories(neuron PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/generated third_party)
target_link_libraries(neuron PRIVATE neuron_eval neuron_job Threads::Threads)
if(WIN32)
    target_link_libraries(neuron PRIVATE ws2_32)
endif()

# Unit test: engine's binormal (Wickens) statistical ROC vs theory.
# Links only the statistics core — no GSL needed.
enable_testing()
add_executable(check_az
    tests/binormal/check_az.cpp)
target_link_libraries(check_az PRIVATE neuron_core)
add_test(NAME binormal_az COMMAND check_az)

# Literature acceptance test: the engine's binormal ROC against Wickens' own
# published worked example (Table 5.1, analysed by hand through chapter 5).
# Self-verifying against the literature, like the Hosmer-Lemeshow log likelihood.
add_executable(check_wickens
    tests/binormal/check_wickens.cpp)
target_link_libraries(check_wickens PRIVATE neuron_core)
add_test(NAME binormal_wickens COMMAND check_wickens)

# Unit test: the Matrix row-gather primitive (includerows -- repeats and
# arbitrary order allowed, bounds throw) and includecols' output dimension.
add_executable(check_matrix
    tests/matrix/check_matrix.cpp)
target_link_libraries(check_matrix PRIVATE neuron_core)
add_test(NAME matrix_gather COMMAND check_matrix)

# Regression test: Hosmer-Lemeshow on datasets past 10,000 rows (fixed-size
# stack arrays segfaulted there until 2026-07-16). No GSL needed.
add_executable(check_hl
    tests/twoset/check_hl.cpp)
target_link_libraries(check_hl PRIVATE neuron_core)
add_test(NAME hosmer_lemeshow_large COMMAND check_hl)

# Unit test: the plateau early-stop detector (ROADMAP 2 Phase 3). Header-only
# and engine-free -- five synthetic error traces. No GSL needed.
add_executable(check_plateau
    tests/plateau/check_plateau.cpp)
target_include_directories(check_plateau PRIVATE src)
add_test(NAME plateau_detector COMMAND check_plateau)

# Unit test: engine output is capturable via util::set_screen() (the
# GUI/server contract). No GSL needed.
add_executable(check_capture
    tests/capture/check_capture.cpp)
target_link_libraries(check_capture PRIVATE neuron_data)
add_test(NAME screen_capture COMMAND check_capture)

# Unit test: the strict request-field parsers (ROADMAP 4 item B9) -- the
# whitespace rule, the boolean token set, sign and overflow policy, the
# non-finite refusal, underflow acceptance, and the rule that a refused parse
# never writes its destination. Links neuron_core alone; no GSL, no engine.
# It does NOT prove the handlers use them -- tests/gui/strictparse.sh does.
add_executable(check_parse
    tests/parse/check_parse.cpp)
target_link_libraries(check_parse PRIVATE neuron_core)
add_test(NAME strict_field_parsers COMMAND check_parse)

# Unit test: the stratified test-set splitter (ROADMAP 4 Phase 1). Pins exact
# apportionment, stratification, a leakage-free partition, and seed
# reproducibility at the index level, PLUS a DataSet::randomize integration
# pass (correct row gather, training-set-limit normalization of both sets,
# loaded-state flags, and the all-test / zero-test boundaries). No GSL.
add_executable(check_split
    tests/split/check_split.cpp)
target_link_libraries(check_split PRIVATE neuron_data)
add_test(NAME split_stratified COMMAND check_split)

# Unit test: the OBD hidden-layer resizing primitives AND the grow-then-prune
# sizing driver (ROADMAP 2 Phase 4). Trains real SimpleProp networks and runs
# the driver (which clones via netclone and probes via autoalgo), so it links
# the training engine and those helpers (and GSL, which network.cpp references).
add_executable(check_obd
    tests/obd/check_obd.cpp)
target_link_libraries(check_obd PRIVATE neuron_engine)
add_test(NAME obd_sizing COMMAND check_obd)

# CHARACTERIZATION test: what SimpleProp and BareProp observably do, pinned
# BEFORE the shared-mechanism extraction (docs/refactor_audit.md 8.2) so that
# extraction can be proven behavior-preserving. Asserts forward outputs, exact
# errors on the batch/non-batch, auto-step-size and CGD/Shanno paths, the saved
# file format, a save/load round trip, and validation-set monitoring for all
# four model types. Run with --capture to print the literals it compares
# against; re-capturing is not a fix (see the header).
add_executable(check_props
    tests/props/check_props.cpp)
target_link_libraries(check_props PRIVATE neuron_engine)
add_test(NAME prop_characterization COMMAND check_props)

# CHARACTERIZATION test: the batch objective and the RAW batch gradient, pinned
# TOGETHER before the Phase 3 extraction moves them into one authoritative
# evaluation that L-BFGS can also call at a trial point. Nothing pinned the
# relation between them: every gradient component of SimpleProp and BareProp is
# checked against a central finite difference of the objective the engine
# itself reports, over LMS/cross-entropy and decay on/off; BackProp's gradient
# is recovered from the update it performs and checked the two ways its private
# weights allow. No captured literals -- every assertion is a same-process
# relation, so it holds on any platform with correctly rounded exp/log.
add_executable(check_batchgradient
    tests/network/check_batchgradient.cpp)
target_link_libraries(check_batchgradient PRIVATE neuron_engine)
add_test(NAME network_batch_gradient COMMAND check_batchgradient)

# Unit test: the packed parameter boundary the Phase 3 L-BFGS work needs -- the
# weights out and back in one layout, and f/raw g at the installed weights
# without training. Asserts the round trip through a PERTURBED vector (an
# implementation ignoring its argument survives an identity round trip), that
# the boundary's objective is BIT-IDENTICAL to the legacy training pass at the
# same weights, that an evaluation moves no weight and no optimizer state,
# element-by-element finite differences for all three neural models, and that a
# model without the boundary refuses rather than returning a plausible value.
add_executable(check_packedboundary
    tests/network/check_packedboundary.cpp)
target_link_libraries(check_packedboundary PRIVATE neuron_engine)
add_test(NAME network_packed_boundary COMMAND check_packedboundary)

# Unit test: DeLong's test for correlated ROC areas (ROADMAP 4 Phase 4, the
# locked-test inference layer). AUC == the engine's exact Mann-Whitney area, a
# 2-classifier fixture pinned to an independent Python DeLong, and structural
# identities (identical classifiers -> Var(delta)==0; degeneracy refused). Links
# the twoset stack for the Mann-Whitney cross-check; no GSL.
add_executable(check_delong
    tests/delong/check_delong.cpp)
target_link_libraries(check_delong PRIVATE neuron_infer)
add_test(NAME delong_areas COMMAND check_delong)

# Literature acceptance test: Obuchowski's nonparametric CLUSTERED ROC-area
# covariance (ROADMAP 4). Held to the published worked example -- areas,
# standard errors, difference, interval, p -- AND to the nine intermediate
# structural components printed in that source's table footer, which is what
# distinguishes an acceptance test from a coincidence (several wrong formulas
# land near the right SE). Also pins the reduction toward ordinary DeLong on
# singleton clusters, a paired whole-cluster bootstrap cross-check, and the
# degeneracy/refusal contracts. Links delong for the side-by-side comparison.
add_executable(check_clustered
    tests/clustered/check_clustered.cpp)
target_link_libraries(check_clustered PRIVATE neuron_infer)
add_test(NAME clustered_auc COMMAND check_clustered)

# A reproducible SCALE probe for the partition + clustered-inference layer, on
# generated data. Built (so it cannot rot) but NOT registered with add_test: it
# measures wall time and peak memory, which are machine-dependent, and a timing
# assertion in the suite would be a flake generator. Run it deliberately:
#   ./build/scale_probe [rows] [clusters]
add_executable(scale_probe
    tests/clustered/scale_probe.cpp)
# Spans two SIBLING layers: the clustered estimator (infer) and the group fold
# planner (data). Neither sits above the other, so it names both.
target_link_libraries(scale_probe PRIVATE neuron_infer neuron_data)

# Unit test: the generic cross-validation runner (ROADMAP 4 Phase 4b-CV). Runs a
# train procedure over a stratified fold plan and checks every row gets one
# out-of-fold prediction, the pooled OOF AUC beats chance, and it is reproducible.
add_executable(check_crossval
    tests/crossval/check_crossval.cpp)
target_link_libraries(check_crossval PRIVATE neuron_eval)
add_test(NAME crossval_runner COMMAND check_crossval)

# Unit test: stopping conditions are evaluated independently of REPORTING
# CADENCE (2026-07-26). Iterative::train() used to recalculate the maximum
# absolute gradient only inside the print block, so logarithmic vs linear
# printing moved the stopping iteration -- and a ceiling landing between the
# real crossing and the next print point reported a false failure to converge.
# A scripted Iterative double pins the crossing; seeded SimpleProp runs pin
# real-model invariance across all three optimizers (GSL is linked for those).
add_executable(check_gradcadence
    tests/iterative/check_gradcadence.cpp)
target_link_libraries(check_gradcadence PRIVATE neuron_engine)
add_test(NAME iterative_grad_cadence COMMAND check_gradcadence)

# A clone must never inherit the quiet flag -- "not copied" implemented as an
# explicit assignment, not as an omission that reads indeterminate memory.
add_executable(check_quietcopy
    tests/iterative/check_quietcopy.cpp)
target_link_libraries(check_quietcopy PRIVATE neuron_engine)
add_test(NAME iterative_quiet_copy COMMAND check_quietcopy)

# Regression test: a QUIET run is a SILENT run, never a DIFFERENT one.
# Network::runHeader() computed regularizer and decayTerm -- two constants the
# TRAINING math reads -- and Iterative::train() called it inside its quiet
# guard, so a quiet run with weight decay on trained on uninitialised doubles.
# Reachable by a fresh model trained quietly, or a clone of a template that
# never trained; NOT by the known stepwise path, whose candidates clone an
# already-trained model and inherit both constants through Network::copy.
# Asserts that quiet and audible runs agree exactly on error, weights, stop
# reason, convergence and iteration count. Found 2026-08-01 by tests/props.
add_executable(check_quietprep
    tests/iterative/check_quietprep.cpp)
target_link_libraries(check_quietprep PRIVATE neuron_engine)
add_test(NAME iterative_quiet_prepare COMMAND check_quietprep)

# Regression test: the condition number, and the rule that reporting must not
# move the model. The gradient-harvesting mechanism this originally guarded is
# gone -- the diagnostic is now the unpenalized observed Fisher information
# X'VX, shared with the Wald covariance. What it guards now:
#   * reporting leaves the fitted and optimizer state untouched, so a continued
#     run matches a control that never reported (canonical, CGD and Shanno);
#   * the reported ratio and both extreme eigenvalues agree with an independent
#     analytic fixture -- one input makes X'VX 2x2, whose eigenvalues are
#     closed-form, computed there without Matrix::dotprod or GSL;
#   * weight decay cannot move the design condition number when the fit is held
#     fixed, which is the property that makes it a collinearity diagnostic.
add_executable(check_condnum
    tests/iterative/check_condnum.cpp)
target_link_libraries(check_condnum PRIVATE neuron_engine)
add_test(NAME iterative_condition_number COMMAND check_condnum)

# Regression test: weight decay is a property of lambda and eta, not of N (D4).
# The canonical decay multiplier sat above the batch/on-line split, so it ran
# once per EXEMPLAR in both modes -- giving a per-epoch factor of
# (1-eta*decay)^N in batch mode, exponential in the dataset size. Asserts that
# duplicating every observation (which leaves the mean gradient identical)
# leaves the fit identical, for all four models.
add_executable(check_decay
    tests/iterative/check_decay.cpp)
target_link_libraries(check_decay PRIVATE neuron_engine)
add_test(NAME iterative_weight_decay COMMAND check_decay)

# Unit test: the output error function, single and multiple output. Pins the
# LMS and cross-entropy values, the cross-entropy boundary approximations, and
# the identity that nothing previously checked -- that the multi-output
# constructor is the single-output one summed over components. Also covers the
# sumSquaredDifference primitive the LMS branch uses. Header-only; links the
# numerical core.
add_executable(check_errorfunc tests/errorfunc/check_errorfunc.cpp)
target_link_libraries(check_errorfunc PRIVATE neuron_core)
add_test(NAME error_function COMMAND check_errorfunc)

# Unit test: TwoSet's four classification rates and their zero denominators.
# Each computed the ratio and THEN checked whether the denominator was zero --
# a guard running after the thing it guards against (a double over zero does
# not trap, so the throw still happened, but only after forming an infinity).
# Asserts all four zero-denominator throws, exact ordinary ratios against a
# hand-built confusion matrix, and the unchanged no-threshold contract.
add_executable(check_rates tests/twoset/check_rates.cpp)
target_link_libraries(check_rates PRIVATE neuron_core)
add_test(NAME twoset_rates COMMAND check_rates)

# The bounds contract of the numerical vector layer (D5).
#
# NDEBUG is set EXPLICITLY, not inherited. Every precondition in vector_ops.h
# was an assert, and this project forces Release (-O3 -DNDEBUG) when no build
# type is given -- so all nineteen ctest targets already ran with assertions
# compiled out and not one of those assertions had ever executed in the gate
# chain. A checked build would let this target pass while leaving the shipped
# defect intact, so it must not depend on how the build directory was
# configured.
#
# Asserts the exception contract of every operation whose precondition, if
# violated, can read or write outside a container -- and the positive controls
# that the successful path is unchanged, including the deliberate PREFIX
# semantics of the compound operators (a longer right operand is legal and the
# bias-slot arithmetic of SimpleProp and BackProp depends on it).
#
# tests/vectorops/run_bounds_demo.sh runs each case in its own process. That is
# a demonstration driver for use against sabotaged or historical builds, where
# a case that takes a SIGBUS would otherwise hide every case after it.
# Regression test: a fresh iterative model has a DEFINED iteration count.
#
# Iterative's constructor set nineteen members and not `iteration`. Not a
# training regression -- train()'s loop assigns it before the first trainSet(),
# and that path is byte-identical -- but reachable through the public object
# contract: getIterations() on an untrained model (RegressNet reads it), copies
# of fresh objects, and a direct trainSet() call, where Network::engine()
# branches on the counter and an indeterminate value took the
# conjugate-direction branch with lastF/lastG still empty. D5's bounds check
# turned that silent out-of-bounds read into a thrown exception, which is how it
# was found.
add_executable(check_iterinit tests/iterative/check_iterinit.cpp)
target_link_libraries(check_iterinit PRIVATE neuron_engine)
add_test(NAME iterative_iteration_init COMMAND check_iterinit)

# Characterization test: SimpleProp and BareProp, before they share a base.
#
# The two one-hidden-layer models carry byte-identical state and six methods
# differing only by the class name (docs/refactor_audit.md 8.2 category A). Holds
# still what must NOT move when those go to an intermediate base: two distinct
# runtime types and objType, saved-file bytes and their first line, the factory
# dispatch, df() and outputHeader(), and -- the subtle one -- that the bias
# ARCHITECTURE is fixed at construction and is not Network::biasFlag, which is
# publicly settable. Cross-type assignment is refused at COMPILE time, since a
# runtime test cannot catch a program that compiles.
add_executable(check_onehidden tests/onehidden/check_onehidden.cpp)
target_link_libraries(check_onehidden PRIVATE neuron_engine)
add_test(NAME onehidden_models COMMAND check_onehidden)

# Characterization test: the automatic step-size search, in all four models.
#
# SimpleProp, BareProp, BackProp and Logistic each carry their own copy of the
# eta search at the top of trainSet(). Pinned BEFORE consolidation. Portable by
# construction -- innerTrainSet() call counts and the selected eta (a small
# power of gamma), never multi-iteration floating point, which check_bpoptimizer
# proved is not portable across AppleClang/GCC/MSVC. Restoration is asserted
# without literals: a run that SEARCHED and settled on eta* must land where a
# run that never searched and was GIVEN eta* lands, for all three optimizers.
# Also pins the guard asymmetry -- Logistic searches on autoStep alone, the
# three nets need batch too -- including the Logistic on-line case that no
# interface can reach.
add_executable(check_autostep tests/network/check_autostep.cpp)
target_link_libraries(check_autostep PRIVATE neuron_engine)
add_test(NAME network_auto_step_size COMMAND check_autostep)

# Characterization test: the last-operation file, from all three writers.
#
# Iterative::train(), LDFA::train() and QDFA::train() each opened
# util::run_path( lastopFilename ) with ios::trunc and wrote the run's file
# stream into it. Model owns lastopFlag and lastopFilename, so Model owns the
# write. Asserts the file contents, the DISABLED case leaving an existing file
# alone, the unopenable-path message (which must not abort the run), and
# TRUNCATION -- the property an extraction most easily loses and a reader would
# not notice for months, because model.txt would simply keep growing.
add_executable(check_writelastop tests/model/check_writelastop.cpp)
target_link_libraries(check_writelastop PRIVATE neuron_engine)
add_test(NAME model_write_lastop COMMAND check_writelastop)

# Characterization test: every way Iterative::train() can stop, and what it says.
#
# Six stopping conditions published themselves with the same five-line block --
# compose a message, write it to the log and screen unless quiet, record the
# reason, break. Pins the StopReason, its machine-readable token, converged(),
# the exact message, and the quiet contract for all six, PLUS the observer's
# three distinct reasons and the iteration ceiling (which is not a stopping
# condition and reports through the epilogue instead). A scripted Iterative, so
# each exit is reached in isolation rather than racing a real model's rules.
add_executable(check_announcestop tests/iterative/check_announcestop.cpp)
target_link_libraries(check_announcestop PRIVATE neuron_engine)
add_test(NAME iterative_announce_stop COMMAND check_announcestop)

# Characterization + regression test: DataSet::metricsReport, BOTH halves.
#
# The training and test blocks were the same forty lines written twice. A golden
# transcript proves the bytes it contains did not move; it does not prove which
# branch produced them, and a report run without a test set exercises only one.
# This drives each half explicitly, plus the four division-by-zero catch blocks
# (half the duplicated body) and the two refusals, so the extraction can be shown
# to change nothing a reader sees. No GSL needed.
add_executable(check_metricsreport tests/dataset/check_metricsreport.cpp)
target_link_libraries(check_metricsreport PRIVATE neuron_data)
add_test(NAME dataset_metrics_report COMMAND check_metricsreport)

# Regression test: the chosen optimizer must reach BackProp's weights.
#
# LEGACY BUG #12. In batch/epoch mode BackProp::innerTrainSet() copied the
# averaged gradient into Gradient, let engine() turn it into the selected search
# direction, and then updated the weights from WeightsAccumulate -- which the
# optimizer never touches. Batch BackProp under CGD or Shanno was plain gradient
# descent announcing itself as something else. SimpleProp, BareProp and on-line
# BackProp all update from the post-engine() gradient and were correct.
#
# Nothing caught it because tests/props' CGD/Shanno case covers SimpleProp and
# BareProp only and no golden fixture uses BackProp. Pins the invariants the
# correction must not move (canonical batch, all three on-line paths) alongside
# the corrected behavior.
add_executable(check_bpoptimizer tests/backprop/check_bpoptimizer.cpp)
target_link_libraries(check_bpoptimizer PRIVATE neuron_engine)
add_test(NAME backprop_optimizer COMMAND check_bpoptimizer)

add_executable(check_vector_bounds tests/vectorops/check_vector_bounds.cpp)
target_link_libraries(check_vector_bounds PRIVATE neuron_core)
target_compile_definitions(check_vector_bounds PRIVATE NDEBUG)
add_test(NAME vector_ops_bounds COMMAND check_vector_bounds)

# The Matrix bounds and dimension contract (D9), asserted in the build we ship
# (NDEBUG, as above). 55 cases: 14 BoundsViolation, 32 DimensionMismatch, 4
# BadSize, 1 Singular, 4 deliberately legal -- including the short-destination
# prefix rule of dotprod and both empty column selections. It was written and
# run BEFORE the policy existed, where 44 of its cases had no contract to catch
# them (39 silent, 4 fatal, 1 throwing incidentally); that run is the red
# evidence, recorded in docs/refactor_audit.md section 12.6, and this add_test line
# is what changed when the policy landed. tests/matrix/run_matrix_bounds_demo.sh
# still runs each case in its own process, which is how a case that CRASHES
# rather than returning is attributed to itself.
# CHARACTERIZATION test: what LDFA and QDFA observably do, pinned BEFORE the
# reporting-scaffold extraction (docs/refactor_audit.md 2.4/2.5). Reports, graded
# guesses, ROC availability per output arity, the DIRECTION of each
# discriminant (larger wins for the linear, smaller for the quadratic -- the
# thing a shared "which one wins" flag would risk), save behavior, history and
# last-operation behavior, the singular and non-discrete refusals, and that a
# standalone analysis leaves a trained network's guesses alone. Trains a real
# SimpleProp for that last one, so it links the engine.
add_executable(check_dfa tests/dfa/check_dfa.cpp)
target_link_libraries(check_dfa PRIVATE neuron_engine)
add_test(NAME dfa_characterization COMMAND check_dfa)

# CHARACTERIZATION of stepwise regression before reverse_regress() and
# forward_regress() share anything (docs/refactor_audit.md item 1.3, section 15).
# 83 of reverse's 128 executable lines are byte-identical to forward's; the
# other 45 are the published direction semantics -- opposite nesting, opposite
# Wilks argument order, opposite winner sense, and a trained empty baseline that
# only forward has. This pins those, plus the audit trail's record-before-judge
# rule, the convergence contract on candidates, grouped variables, the LMS
# refusal, and that the source network is never mutated. Trains real Logistic
# and SimpleProp models, so it links the engine.
add_executable(check_regressnet tests/regressnet/check_regressnet.cpp)
target_link_libraries(check_regressnet PRIVATE neuron_engine)
add_test(NAME regressnet_characterization COMMAND check_regressnet)

# The asynchronous-job lifecycle and the CLI process boundary. Links only
# neuron_job (plus neuron_core, for the Matrix contract exception the CLI
# boundary was built to report) -- no GSL, no engine, so it runs in a second.
# It exists because the two ordering invariants below are unreachable through
# HTTP; see docs/refactor_audit.md section 20.4 for the measurement.
add_executable(check_asyncjob tests/asyncjob/check_asyncjob.cpp)
target_link_libraries(check_asyncjob PRIVATE neuron_job neuron_core)
add_test(NAME asyncjob_lifecycle COMMAND check_asyncjob)

add_executable(check_matrix_bounds tests/matrix/check_matrix_bounds.cpp)
target_link_libraries(check_matrix_bounds PRIVATE neuron_core)
target_compile_definitions(check_matrix_bounds PRIVATE NDEBUG)
add_test(NAME matrix_bounds COMMAND check_matrix_bounds)

# --- the optimizer benchmark harness (optimizer research, Phase 0) ----------
#
# Two targets over one shared header, tests/optimizer/harness.h, because the
# binary that MEASURES and the binary that proves the measurement trustworthy
# must run the same code or the proof is about something else.
#
# BINARY IDENTITY. A benchmark row must identify the code that produced it, and
# a revision alone cannot: the harness is developed uncommitted, so every row
# would claim a commit whose tree does not contain the binary's own source.
#
#   OPTBENCH_SOURCE_ID  THE AUTHORITY: a hash over every source input capable of
#                       changing what is measured -- the harness AND the engine
#                       the harness links AND this build file. An earlier version
#                       hashed only the three harness files, which meant a dirty
#                       edit to src/network.cpp or src/backprop.cpp could change
#                       the measured optimizer while the id stood still. An
#                       identity that cannot see the thing being benchmarked is
#                       not an identity of the benchmark.
#   OPTBENCH_REV        the committed revision, for orientation.
#   OPTBENCH_GIT_DIRTY  whether the tree differed from that revision AT CONFIGURE
#                       TIME. It is a hint, not the authority -- see below.
#
# EVERY hashed file is also a CMAKE_CONFIGURE_DEPENDS, so editing any of them
# re-runs cmake, refreshes the id, and rebuilds the targets together. Without
# that the id would be a snapshot and a rebuilt binary could keep advertising
# source it no longer contains.
#
# .git/HEAD and .git/index join the dependency list so that COMMITTING -- which
# changes rev and clears dirty without touching a single source file -- also
# refreshes them. That closes the staleness Sol identified. What remains, and is
# documented rather than overclaimed: editing a file OUTSIDE this list (a doc,
# say) can leave `dirty` reading 0 until the next configure. That cannot change
# what is measured, which is exactly why source_id and not dirty is the
# authority.
file(GLOB OPTBENCH_ENGINE_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
set(OPTBENCH_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/optimizer/harness.h
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/optimizer/optimizer_probe.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/optimizer/check_optimizer_harness.cpp
    # prepare_data.py decides WHAT DATA the Step 0B workloads run on. It is not
    # compiled, but changing it changes what is measured just as surely as
    # changing the engine does, so it belongs to the same identity. (The data
    # itself is identified per row by content hash -- data_id -- so provenance
    # does not depend on this directory surviving.)
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/optimizer/prepare_data.py
    ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
    ${OPTBENCH_ENGINE_SOURCES})
list(SORT OPTBENCH_SOURCES)     # deterministic order, so the id is reproducible
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${OPTBENCH_SOURCES})
foreach(_gitfile HEAD index)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/.git/${_gitfile})
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/.git/${_gitfile})
    endif()
endforeach()

# THE ENGINE IDENTITY, separately from the benchmark's own.
#
# The committed endpoint table in harness.h holds MEASUREMENTS of how the engine
# behaves. Writing a measurement into harness.h necessarily changes
# OPTBENCH_SOURCE_ID -- so "the targets and the arms share a source identity" is
# a condition that can never hold, and demanding it would mean re-characterizing
# forever. What the targets actually depend on is src/, and nothing in the
# harness can change how the engine trains. OPTBENCH_ENGINE_ID hashes exactly
# that, so a characterization can say WHICH ENGINE it describes and a later
# harness edit leaves that claim intact -- while an edit to src/network.cpp
# invalidates it loudly, which is the case that matters.
set(OPTBENCH_ENGINE_BLOB "")
foreach(f ${OPTBENCH_ENGINE_SOURCES})
    if(EXISTS ${f})
        file(SHA256 ${f} _h)
        file(RELATIVE_PATH _rel ${CMAKE_CURRENT_SOURCE_DIR} ${f})
        string(APPEND OPTBENCH_ENGINE_BLOB "${_rel}:${_h}\n")
    endif()
endforeach()
string(SHA256 OPTBENCH_ENGINE_FULL "${OPTBENCH_ENGINE_BLOB}")
string(SUBSTRING "${OPTBENCH_ENGINE_FULL}" 0 16 OPTBENCH_ENGINE_ID)
list(LENGTH OPTBENCH_ENGINE_SOURCES OPTBENCH_ENGINE_COUNT)

set(OPTBENCH_SOURCE_BLOB "")
foreach(f ${OPTBENCH_SOURCES})
    if(EXISTS ${f})
        file(SHA256 ${f} _h)
        # The PATH is hashed with the content, so moving a file between names
        # changes the id even when the bytes do not.
        file(RELATIVE_PATH _rel ${CMAKE_CURRENT_SOURCE_DIR} ${f})
        string(APPEND OPTBENCH_SOURCE_BLOB "${_rel}:${_h}\n")
    endif()
endforeach()
string(SHA256 OPTBENCH_SOURCE_FULL "${OPTBENCH_SOURCE_BLOB}")
string(SUBSTRING ${OPTBENCH_SOURCE_FULL} 0 16 OPTBENCH_SOURCE_ID)
list(LENGTH OPTBENCH_SOURCES OPTBENCH_SOURCE_COUNT)

execute_process(COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE OPTBENCH_REV
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(NOT OPTBENCH_REV)
    set(OPTBENCH_REV "unknown")
endif()

execute_process(COMMAND git status --porcelain
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE OPTBENCH_STATUS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
if(OPTBENCH_STATUS STREQUAL "")
    set(OPTBENCH_DIRTY 0)
else()
    set(OPTBENCH_DIRTY 1)   # also the safe default when git is unavailable
endif()
message(STATUS "optimizer benchmark identity: rev=${OPTBENCH_REV} dirty=${OPTBENCH_DIRTY} "
    "source_id=${OPTBENCH_SOURCE_ID} over ${OPTBENCH_SOURCE_COUNT} source files; "
    "engine_id=${OPTBENCH_ENGINE_ID} over ${OPTBENCH_ENGINE_COUNT} engine files")

# The MEASURING end. Built so it cannot rot; deliberately NOT registered with
# add_test, because wall time is machine-dependent and a timing assertion in the
# suite is a flake generator (the scale_probe precedent). Run it deliberately:
#   ./build/optimizer_probe --all
add_executable(optimizer_probe tests/optimizer/optimizer_probe.cpp)
# neuron_eval, not neuron_engine: the Step 0B repeated-fit workload measures the
# real cross-validation runner and its locked refit, so it links the layer that
# owns them rather than reimplementing a fold loop the user never runs.
target_link_libraries(optimizer_probe PRIVATE neuron_eval)
target_include_directories(optimizer_probe PRIVATE tests/optimizer)
target_compile_definitions(optimizer_probe PRIVATE
    OPTBENCH_GIT_REV="${OPTBENCH_REV}"
    OPTBENCH_GIT_DIRTY=${OPTBENCH_DIRTY}
    OPTBENCH_SOURCE_ID="${OPTBENCH_SOURCE_ID}"
    OPTBENCH_SOURCE_COUNT=${OPTBENCH_SOURCE_COUNT}
    OPTBENCH_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    OPTBENCH_ENGINE_ID="${OPTBENCH_ENGINE_ID}"
    OPTBENCH_ENGINE_COUNT=${OPTBENCH_ENGINE_COUNT}
    OPTBENCH_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/optimizer/data")

# The MECHANICS, asserted deterministically and with no timing assertion at all:
# identical starts, the non-vacuity of that identity, split identity, the
# ceiling-is-not-convergence contract, the matched-target verdict, the full-pass
# counter seeing the step-size search, the row schema, refusal by name, and
# leaving no artifacts. This one IS a ctest case, so the harness cannot rot
# silently while the probe still compiles.
add_executable(check_optimizer_harness tests/optimizer/check_optimizer_harness.cpp)
target_link_libraries(check_optimizer_harness PRIVATE neuron_eval)
target_include_directories(check_optimizer_harness PRIVATE tests/optimizer)
target_compile_definitions(check_optimizer_harness PRIVATE
    OPTBENCH_GIT_REV="${OPTBENCH_REV}"
    OPTBENCH_GIT_DIRTY=${OPTBENCH_DIRTY}
    OPTBENCH_SOURCE_ID="${OPTBENCH_SOURCE_ID}"
    OPTBENCH_SOURCE_COUNT=${OPTBENCH_SOURCE_COUNT}
    OPTBENCH_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    OPTBENCH_ENGINE_ID="${OPTBENCH_ENGINE_ID}"
    OPTBENCH_ENGINE_COUNT=${OPTBENCH_ENGINE_COUNT}
    OPTBENCH_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/optimizer/data")
add_test(NAME optimizer_harness COMMAND check_optimizer_harness)

# The ORCHESTRATION layer is part of the evidence system, so it is tested too:
# run_probe.py --self-test feeds synthetic rows through the same validation the
# real campaign uses and requires each malformed, inconsistent or unfair set to
# be REFUSED. Deterministic, no timing, no subprocess, no network.
add_test(NAME optimizer_runner_selftest
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/optimizer/run_probe.py --self-test)
