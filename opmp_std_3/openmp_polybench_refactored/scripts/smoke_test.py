"""
smoke_test.py -- validates that
  1. The metadata format survives fgetting written by bash/C and read by bench_io.
  2. The specific error seen on Falkor's side
     ('Error tokenizing data. C error: Expected 3 fields in line 3, saw 15')
     is reproduced from the OLD parser and fixed by the NEW one.
  3. All 15 Julia-compatible columns are present and correctly typed.

Run from the project root:
    python3 scripts/smoke_test.py

Exits nonzero if any check fails.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from bench_io import load_csv, load_many, parse_metadata_block, run_header_line


# Intentionally awful: CPU string has commas and parens; compiler has spaces;
# every chars that used to break CSV parsing are present somewhere.
# The C / bash side is supposed to sanitize these BEFORE writing; we emulate
# what a correctly-sanitized metadata block looks like.
CSV_SAMPLE = """\
# schema=1; language=openmp; host=laptop-0gogn00r; os=Linux-6.8.0; cpu=Intel(R)_Core(TM)_i5-10210U_CPU_@_1.60GHz; cores_logical=8; cores_physical=4; compiler=gcc; compiler_version=13.2.0; compiler_flags=-O3_-fopenmp; openmp=201511; run_date=2026-04-17T12:14:41+0200; project=openmp_polybench_refactored; git_commit=a1b2c3d4e5f6
# benchmark=heat3d; dataset=MEDIUM; threads=8; iterations=7; warmup=3; mode=full; omp_proc_bind=close; omp_places=cores; omp_schedule=unset
benchmark,dataset,strategy,threads,is_parallel,min_ms,median_ms,mean_ms,std_ms,gflops,speedup,efficiency_pct,verified,max_error,allocations
heat3d,MEDIUM,sequential,1,false,142.1230,143.4500,143.8100,1.2300,3.50,1.00,,PASS,0.00e+00,0
heat3d,MEDIUM,threads_static,1,true,141.8900,142.7700,143.0200,1.1500,3.51,1.00,100.00,PASS,0.00e+00,0
heat3d,MEDIUM,threads_static,2,true,72.4100,73.0100,73.3300,0.8200,6.88,1.96,97.93,PASS,0.00e+00,0
heat3d,MEDIUM,threads_static,4,true,38.2200,38.9000,39.2100,0.7100,13.03,3.72,92.95,PASS,0.00e+00,0
heat3d,MEDIUM,threads_static,8,true,22.9100,23.5500,23.9800,0.6300,21.74,6.20,77.52,PASS,0.00e+00,0
"""

EXPECTED_COLUMNS = {
    "benchmark", "dataset", "strategy", "threads", "is_parallel",
    "min_ms", "median_ms", "mean_ms", "std_ms", "gflops",
    "speedup", "efficiency_pct", "verified", "max_error", "allocations",
}


def _check(name, cond, detail=""):
    mark = "OK  " if cond else "FAIL"
    print(f"[{mark}] {name}" + (f"  :: {detail}" if detail else ""))
    if not cond:
        _check.failed = True
_check.failed = False


def test_old_parser_reproduces_bug(path: Path) -> None:
    """Pandas without comment='#' must fail with the exact observed error."""
    try:
        pd.read_csv(path)
    except pd.errors.ParserError as e:
        msg = str(e)
        _check("old parser raises ParserError",
               "Expected" in msg and "fields" in msg and "saw 15" in msg,
               detail=msg.splitlines()[0])
        return
    _check("old parser raises ParserError", False,
           detail="did NOT raise -- the pre-fix bug is gone, nothing to fix?")


def test_metadata_parse(path: Path) -> None:
    meta = parse_metadata_block(path)
    _check("metadata: schema present", meta.get("schema") == "1")
    _check("metadata: host carries through", meta.get("host") == "laptop-0gogn00r")
    _check("metadata: cpu survives special chars",
           meta.get("cpu", "").startswith("Intel(R)_Core"))
    _check("metadata: run_date is ISO-ish",
           "T" in meta.get("run_date", "") and "2026" in meta.get("run_date", ""))
    _check("metadata: openmp version present",
           meta.get("openmp") == "201511")
    _check("metadata: iterations + warmup present",
           meta.get("iterations") == "7" and meta.get("warmup") == "3")
    _check("metadata: omp_proc_bind captured",
           meta.get("omp_proc_bind") == "close")


def test_load_csv(path: Path) -> None:
    df = load_csv(path)
    _check("load_csv: returns DataFrame", df is not None and not df.empty)
    if df is None or df.empty:
        return
    got = set(df.columns)
    missing = EXPECTED_COLUMNS - got
    _check("load_csv: all 15 Julia-compatible columns present",
           not missing, detail=f"missing={missing}")
    _check("load_csv: is_parallel coerced to bool",
           df["is_parallel"].dtype == bool,
           detail=f"dtype={df['is_parallel'].dtype}")
    _check("load_csv: 5 data rows", len(df) == 5, detail=f"len={len(df)}")
    _check("load_csv: metadata attached", bool(df.attrs.get("meta")))
    _check("load_csv: host column promoted",
           "host" in df.columns and df["host"].iloc[0] == "laptop-0gogn00r")
    _check("load_csv: run_date column promoted", "run_date" in df.columns)
    # Speedup sanity: sequential row should be 1.0
    seq_row = df[(df["strategy"] == "sequential")]
    _check("load_csv: sequential speedup == 1.0",
           not seq_row.empty and float(seq_row["speedup"].iloc[0]) == 1.0)
    hdr = run_header_line(df.attrs.get("meta", {}), df)
    _check("load_csv: header line non-empty", bool(hdr), detail=hdr)


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "scaling_heat3d_MEDIUM_20260417_121441.csv"
        p.write_text(CSV_SAMPLE)

        print("==> old parser (bug reproduction)")
        test_old_parser_reproduces_bug(p)
        print("==> metadata parser")
        test_metadata_parse(p)
        print("==> load_csv (the fix)")
        test_load_csv(p)

    return 1 if _check.failed else 0


if __name__ == "__main__":
    sys.exit(main())
