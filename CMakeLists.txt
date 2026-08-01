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
# all built at once (refactor_audit.md section 10).
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
    src/simpleprop.cpp src/bareprop.cpp src/backprop.cpp src/logistic.cpp
    src/dfa.cpp src/ldfa.cpp src/qdfa.cpp
    src/netclone.cpp src/modelfactory.cpp src/regressnet.cpp
    src/autoalgo.cpp src/obd.cpp)
target_link_libraries(neuron_engine PUBLIC neuron_data GSL::gsl GSL::gslcblas)

add_library(neuron_eval STATIC
    src/crossval.cpp src/cvadapters.cpp src/cvreport.cpp)
target_link_libraries(neuron_eval PUBLIC neuron_engine neuron_infer)

# The driver and the embedded GUI. gui.cpp is the only translation unit that
# needs the generated page header and the vendored httplib, so those include
# directories stay PRIVATE to this target rather than leaking into a library.
add_executable(neuron src/neuron.cpp src/gui.cpp)
target_include_directories(neuron PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/generated third_party)
target_link_libraries(neuron PRIVATE neuron_eval Threads::Threads)
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
# BEFORE the shared-mechanism extraction (refactor_audit.md 8.2) so that
# extraction can be proven behavior-preserving. Asserts forward outputs, exact
# errors on the batch/non-batch, auto-step-size and CGD/Shanno paths, the saved
# file format, a save/load round trip, and validation-set monitoring for all
# four model types. Run with --capture to print the literals it compares
# against; re-capturing is not a fix (see the header).
add_executable(check_props
    tests/props/check_props.cpp)
target_link_libraries(check_props PRIVATE neuron_engine)
add_test(NAME prop_characterization COMMAND check_props)

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
add_executable(check_vector_bounds tests/vectorops/check_vector_bounds.cpp)
target_link_libraries(check_vector_bounds PRIVATE neuron_core)
target_compile_definitions(check_vector_bounds PRIVATE NDEBUG)
add_test(NAME vector_ops_bounds COMMAND check_vector_bounds)
