- **2026-07-12 (tools begin)** — `tools/` scaffolded with the **stdlib-only rule**
  (bare python3, no pip/venv — Craig's #1 distribution pain; CI runs the tools with
  system python3 on all 3 OSes to enforce it). First tool: `mkdataset.py`, faithful
  port of mkdataset.pl (blank-outcome row elimination, non-numeric row warnings,
  missing-value indicator pairs "0,0"/"1,value", --key and --inputs files; csv-module
  parsing handles quoted Excel fields, an improvement over Perl's split). Fixture
  test `tests/tools/run_tools.sh` (--bless pattern) diffs 4 outputs against committed
  expected; output verified loadable by the engine (3x5 raw load). Windows note:
  scripts locate python3 OR python; output files written with newline='\n'.
