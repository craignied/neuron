#!/usr/bin/env python3
"""Require the design Manifest to index its public vocabulary.

LaTeX's makeindex can sort only the entries authors supplied.  This check
guards the other half of the contract: every top-level source class and
namespace, the important public result/configuration objects, and a curated
set of principal methods must have a navigable entry in the Manifest source.
"""

from pathlib import Path
import re
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
    "autoalgo!Probe",
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
    },
    "OneHiddenNet": {"setHidden", "randomize", "save", "load", "pack", "innerTrainSet", "propagate"},
    "SimpleProp": {"growHidden", "removeHidden", "hiddenSaliency"},
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
    "autoalgo": {"pick"},
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

    if missing_roots or missing_entries:
        if missing_roots:
            print("Manifest index is missing public classes/namespaces:", file=sys.stderr)
            for name in missing_roots:
                print(f"  {name}", file=sys.stderr)
        if missing_entries:
            print("Manifest index is missing major objects/methods:", file=sys.stderr)
            for name in missing_entries:
                print(f"  {name}", file=sys.stderr)
        return 1

    print(
        f"Manifest index coverage OK: {len(source_roots())} public roots, "
        f"{len(MAJOR_OBJECTS)} major objects, "
        f"{sum(map(len, PRINCIPAL_METHODS.values()))} principal methods"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
