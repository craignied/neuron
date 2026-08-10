#!/usr/bin/env python3
"""Require the design Manifest to index its public vocabulary.

LaTeX's makeindex can sort only the entries authors supplied.  This check
guards the other half of the contract: every top-level source class and
namespace, the important public result/configuration objects, and a curated
set of principal methods must have a navigable entry in the Manifest source.

It also requires the PUBLISHED docs/manifest.pdf to carry every heading the
tex sources declare.  The index half reads only the sources, so it stayed
green on 2026-08-09 when a revision rebuilt the PDF in docs/tex/ but missed
the publish copy -- docs/manifest.pdf shipped without the whole GET
/api/algorithms subsection.  Text is compared after case folding and
stripping non-alphanumerics, so hyphenation, ligatures and line breaks in
the PDF cannot produce a false failure.  Where pdftotext is unavailable
this half is skipped with a printed notice, never silently -- a silent
skip would make it vacuous.  The machines where Manifest work happens have
pdftotext; CI installs poppler where its package manager offers it.
"""

from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TEX_DIR = ROOT / "docs" / "tex"
SRC_DIR = ROOT / "src"


def index_entries(text: str) -> list[str]:
    entries: list[str] = []
    marker = r"\index{"
    start = 0
    while True:
        pos = text.find(marker, start)
        if pos < 0:
            return entries
        depth = 1
        cursor = pos + len(marker)
        entry_start = cursor
        while cursor < len(text) and depth:
            if text[cursor] == "{" and text[cursor - 1] != "\\":
                depth += 1
            elif text[cursor] == "}" and text[cursor - 1] != "\\":
                depth -= 1
            cursor += 1
        if depth:
            raise ValueError(f"unterminated index entry near byte {pos}")
        entries.append(text[entry_start : cursor - 1])
        start = cursor


def canonical(entry: str) -> str:
    """Return the searchable hierarchy, discarding makeindex display markup."""
    levels = []
    for level in entry.split("!"):
        level = level.split("@", 1)[0]
        level = level.replace(r"\_", "_").strip()
        # makeindex sort keys may use prose while the printed term is the
        # source identifier (for example ``clustered auc@\texttt{...}``).
        if level == "clustered auc":
            level = "clustered_auc"
        levels.append(level)
    return "!".join(levels)


def source_roots() -> set[str]:
    declaration = re.compile(r"^(?:class|namespace)\s+([A-Za-z_]\w*)\b")
    roots: set[str] = set()
    for header in SRC_DIR.glob("*.h"):
        for line in header.read_text(encoding="utf-8").splitlines():
            match = declaration.match(line)
            if match:
                roots.add(match.group(1))
    return roots


# Public records that callers name independently of their owning service.
MAJOR_OBJECTS = {
    "DataSet!MonitorSet",
    "TwoSet!ROCfit",
    "TwoSet!CI",
    "Iterative!StopReason",
    "Iterative!Observer",
    "RegressNet!Progress",
    "RegressNet!Candidate",
    "nsplit!Holdout",
    "nsplit!StratHoldout",
    "nsplit!GroupHoldout",
    "nsplit!FoldPlan",
    "nsplit!GroupFoldPlan",
    "evaldesign!Partition",
    "evaldesign!InferenceChoice",
    "evaldesign!SamplingUnit",
    "evaldesign!AucInference",
    "evaldesign!PartitionMethod",
    "auccov!Placements",
    "auccov!Interval",
    "auccov!Contrast",
    "delong!Result",
    "clustered_auc!Result",
    "autoalgo!Settings",
    "autoalgo!Probe",
    "autoalgo!Omission",
    "autoalgo!Result",
    "modelfactory!Spec",
    "obd!ProgressFn",
    "obd!SizeTrial",
    "obd!Config",
    "obd!Eligibility",
    "obd!Result",
    "crossval!Metrics",
    "crossval!FoldResult",
    "crossval!RunResult",
    "crossval!ProcResult",
    "crossval!Progress",
    "crossval!FoldSelection",
    "crossval!ProcedureSpec",
    "crossval!Comparison",
    "crossval!LockedEntry",
    "crossval!LockedResult",
    "crossval!Procedure",
    "crossval!ProgressFn",
    "cvadapters!InnerSplit",
    "cvreport!PlanInfo",
    "cvreport!LockedColumn",
    "cvreport!LockedContrast",
    "cvreport!LockedInfo",
    "cvreport!ArtifactResult",
    "AsyncJob!FailureRenderer",
    "AsyncJob!error series",
    "AsyncJob!result",
    "AsyncJob!OBD progress",
    "AsyncJob!stepwise progress",
    "AsyncJob!cross-validation progress",
}


# Deliberately principal operations, not every convenience accessor/overload.
PRINCIPAL_METHODS = {
    "Matrix": {
        "operator()", "submatrix", "row", "col", "addrow", "addcol",
        "replacerow", "replacecol", "includerows", "includecols",
        "excludecols", "t", "dotprod", "dotprod_row", "outprod", "func",
        "random", "loadfile", "savefile", "resize", "clear", "fill",
        "colsums", "squared", "maxabs", "rowindex", "toVector", "toMatrix", "covariance",
        "inverse", "determinant", "eigenvalues",
        # The symmetric positive-definite normal-equations primitives. Listed
        # because a reader searching the index wants the operation, and because
        # the three share ONE storage contract that only the Manifest states.
        "addOuterUpper", "symmetrize", "solveSPD",
    },
    "Population": {"mean", "var", "std", "skew", "kurtosis"},
    "Funct": {"mnbrak", "brent", "zbrent"},
    "DataSet": {
        "loadRaw", "loadTrain", "loadTest", "randomize", "randomize3",
        "randomize3D", "makeFold", "valLoaded", "getValMatrix", "getNumVal",
        "monitorSet", "monitorSetName", "setStrataColumns", "getStrataColumns",
        "setStrataBins", "getStrataBins", "setGroupColumns", "getGroupColumns",
        "strataKey", "groupKey",
    },
    "TwoSet": {
        "metricsReport", "invalidate", "getTP", "getTN", "getFP", "getFN",
        "getTrapROCarea", "getROCx", "getROCy", "getROCfit", "bootstrapROC",
        "getStatCi", "setBootstrapResamples", "getBootstrapResamples",
        "getStatPoints", "getStatChi2", "getPearsonX2",
    },
    "Model": {"setDataSet", "train", "reportAccuracy", "outputHeader", "getXEerror"},
    "Iterative": {
        "train", "trainSet", "classAccuracy", "getGradMax", "converged",
        "stopReasonToken", "setObserver", "getObserver", "getStopReason",
        "getIterations", "setAutoStop", "getAutoStop", "getAutoStopTol",
        "getAutoStopWindow", "setQuiet", "getQuiet",
    },
    "Network": {
        "forward", "classAccuracy", "searchStepSize", "computeCondNum",
        "sampleTestError", "getCondNum", "getCondMaxEig", "getCondMinEig",
        "packedSize", "packWeights", "unpackWeights", "batchObjectiveGradient",
        "algorithmName", "algorithmLabel", "TRAINING_TYPES",
        "TRAIN_LBFGS",
        "setLBFGSMemory", "getLBFGSMemory", "applyAbsoluteStep",
        "normalEquationsAvailable", "batchNormalEquations", "lmIteration",
        "stepConverged", "TRAIN_IRPROP", "TRAIN_LM", "NormalEvaluator",
    },
    "OneHiddenNet": {
        "setHidden", "randomize", "save", "load", "pack", "innerTrainSet",
        "propagate", "packedSize", "normalEquationsAvailable",
        "batchNormalEquations",
    },
    "SimpleProp": {"growHidden", "removeHidden", "hiddenSaliency"},
    "BackProp": {"packedSize"},
    "Logistic": {"getBetas", "getBetaSE", "getWaldP"},
    "DFA": {"train", "fitDiscriminant"},
    "RegressNet": {
        "setThreshold", "setProgress", "setObserver", "getCandidates",
        "getSelectionPath", "getFinalVariables", "getFitsCompleted", "getComplete",
    },
    "PlateauDetector": {"update", "reset"},
    "nsplit": {"stratifiedHoldout", "groupHoldout", "stratifiedKFold", "stratifiedGroupKFold"},
    "evaldesign": {
        "parseSamplingUnit", "samplingUnitName", "samplingUnitPhrase",
        "independenceStatus", "inferenceName", "inferenceText",
        "inferenceShortName", "partitionMethodName", "describeFolds",
        "describeHoldout", "chooseInference",
    },
    "auccov": {"midranks", "compute", "interval", "contrast"},
    "delong": {"analyze", "interval", "contrast"},
    "clustered_auc": {"analyze", "interval", "contrast"},
    "autoalgo": {
        "Settings", "Probe", "Omission", "Result", "ineligible",
        "candidates", "pick", "DEFAULT_TOTAL_BUDGET_MS",
    },
    "LBFGSObjective": {"currentPoint", "install", "evaluate", "cancelled"},
    "LBFGS": {
        "constructor", "setMemory", "getMemory", "reset", "started", "iterate", "gradMax",
        "objectiveEvaluations", "gradientEvaluations", "lineSearchFailures",
        "historyResets", "curvatureRejections", "cancellations", "pairs",
        "applyInverseHessian", "scaling", "pushPair", "resetHistory",
        "copyConfigurationFrom",
    },
    # iRPROP+ (retained; Network::TRAIN_IRPROP). Listed by method for the
    # same reason LBFGS is: a reader searching the index wants the operation,
    # and computeStep IS the published table.
    # Levenberg-Marquardt (Network::TRAIN_LM). Listed by method for the same
    # reason LBFGS and IRpropState are: a reader searching the index wants the
    # operation, and iterate() IS Algorithm 3.16.
    "LMObjective": {"currentPoint", "install", "evaluateNormal", "cancelled"},
    "LM": {
        "constructor", "published constants", "reset", "started", "iterate",
        "gradMax", "stepConverged", "damping", "rejectionMultiplier",
        "acceptedObjective", "acceptedPoint", "gradient", "lastStep",
        "normalMatrix", "iterations", "acceptances", "rejections",
        "factorizationFailures", "nonFiniteTrials", "cancellations",
    },
    "IRpropState": {
        "constructor", "published constants", "reset", "started", "computeStep",
        "deltas", "previousGradient", "previousStep", "previousObjective",
        "growths", "shrinks", "rollbacks", "heldFlips", "iterations",
        "Ineligible", "NotFinite",
    },
    "modelfactory": {"build", "createByTypeName"},
    "obd": {"classify", "stopToken", "run"},
    "crossval": {"metricsFor", "run", "compare", "evaluateOnce"},
    "cvadapters": {"trainProcedure", "dfaProcedure", "nestedObdProcedure", "innerValidationSplit"},
    "cvreport": {
        "foldPlanText", "rawRowError", "inferenceRan", "splitPlanText",
        "samplingUnitText", "independenceText", "inferenceText",
        "clusterError", "writeArtifacts", "tier1", "tier2", "render",
    },
    "AsyncJob": {
        "constructor", "destructor", "start", "requestStop",
        "clearStopRequest", "joinForShutdown", "isRunning",
        "isStopRequested", "cancelLatch", "clearCvProgress",
        "resetForNewRun", "pushSample", "MAX_POINTS",
    },
    "procguard": {"run"},
    # The strict field parsers (ROADMAP 4 B9). Listed here and not merely as a
    # namespace entry because a caller searching the index is looking for the
    # operation by name -- "how do I read a flag" -- not for "util".
    "util": {
        "parseUnsigned", "parseDouble", "parseBool",
        "unsignedError", "numberError", "boolError", "ParseStatus",
    },
}


def plain_title(title: str) -> str:
    """A heading's text as the reader sees it: markup unwrapped, escapes undone."""
    while True:
        unwrapped = re.sub(r"\\[A-Za-z]+\*?\{([^{}]*)\}", r"\1", title)
        if unwrapped == title:
            break
        title = unwrapped
    title = re.sub(r"\\[A-Za-z]+\s*", " ", title)
    for escaped, literal in ((r"\_", "_"), (r"\&", "&"), (r"\%", "%"),
                             (r"\$", "$"), (r"\#", "#")):
        title = title.replace(escaped, literal)
    return title


def comparable(text: str) -> str:
    """Case-folded alphanumerics only, so LaTeX dashes, ligatures and PDF line
    breaks cannot fake a mismatch."""
    for ligature, expansion in (("\ufb00", "ff"), ("\ufb01", "fi"),
                                ("\ufb02", "fl"), ("\ufb03", "ffi"),
                                ("\ufb04", "ffl")):
        text = text.replace(ligature, expansion)
    return re.sub(r"[^0-9A-Za-z]+", " ", text).lower().strip()


def stale_published_headings(tex: str) -> "list[str] | None":
    """Every declared heading the published docs/manifest.pdf does not carry,
    or None when the check could not run (pdftotext unavailable).

    This is the publish-step guard: latexmk writes docs/tex/manifest.pdf, and
    only a copy makes it the published artifact.  A rebuild whose copy was
    missed leaves every source-reading check green while readers get the old
    document; the PDF's own extracted text is the only place that can fail.
    """
    if shutil.which("pdftotext") is None:
        print(
            "NOTE: pdftotext not found -- the published-PDF staleness half of "
            "this gate was SKIPPED, not passed.",
            file=sys.stderr,
        )
        return None
    published = ROOT / "docs" / "manifest.pdf"
    extracted = subprocess.run(
        ["pdftotext", str(published), "-"], check=True, capture_output=True
    ).stdout.decode("utf-8", "replace")
    haystack = " " + comparable(extracted) + " "
    missing = []
    for match in re.finditer(
        r"\\(?:chapter|section|subsection)\*?\{((?:[^{}]|\{[^{}]*\})*)\}", tex
    ):
        needle = comparable(plain_title(match.group(1)))
        if needle and " " + needle + " " not in haystack:
            missing.append(plain_title(match.group(1)).strip())
    return missing


def main() -> int:
    tex = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(TEX_DIR.glob("*.tex"))
    )
    entries = {canonical(entry) for entry in index_entries(tex)}
    roots = {entry.split("!", 1)[0] for entry in entries}

    missing_roots = sorted(source_roots() - roots)
    required = set(MAJOR_OBJECTS)
    for owner, methods in PRINCIPAL_METHODS.items():
        required.update(f"{owner}!{method}" for method in methods)
    missing_entries = sorted(required - entries)
    stale_headings = stale_published_headings(tex)

    if missing_roots or missing_entries or stale_headings:
        if missing_roots:
            print("Manifest index is missing public classes/namespaces:", file=sys.stderr)
            for name in missing_roots:
                print(f"  {name}", file=sys.stderr)
        if missing_entries:
            print("Manifest index is missing major objects/methods:", file=sys.stderr)
            for name in missing_entries:
                print(f"  {name}", file=sys.stderr)
        if stale_headings:
            print(
                "Published docs/manifest.pdf is STALE -- these tex headings are "
                "absent from its text (rebuild and copy per "
                "docs/manifest_maintenance.md):",
                file=sys.stderr,
            )
            for name in stale_headings:
                print(f"  {name}", file=sys.stderr)
        return 1

    pdf_status = (
        "published-PDF staleness check SKIPPED (no pdftotext)"
        if stale_headings is None
        else "published PDF carries every declared heading"
    )
    print(
        f"Manifest index coverage OK: {len(source_roots())} public roots, "
        f"{len(MAJOR_OBJECTS)} major objects, "
        f"{sum(map(len, PRINCIPAL_METHODS.values()))} principal methods; "
        + pdf_status
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
