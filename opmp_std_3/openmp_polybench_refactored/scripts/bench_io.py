"""
bench_io.py -- shared CSV + metadata I/O for the Julia/OpenMP PolyBench suite.

Every benchmark CSV begins with metadata on one or more '#' comment lines,
then the Julia-compatible header and data rows:

    # schema=1; language=openmp; host=<h>; cpu=<cpu>; cores_logical=<n>; ...
    # benchmark=<name>; dataset=<size>; threads=<n>; iterations=<n>; ...
    benchmark,dataset,strategy,threads,is_parallel,min_ms,median_ms,mean_ms,std_ms,gflops,speedup,efficiency_pct,verified,max_error,allocations
    2mm,LARGE,sequential,1,false,245.1230,...

Rules for metadata values (enforced on the C and bash writer sides):
  - Pair separator:    '; '        (semicolon + space)
  - Key/value sep:     '='
  - Forbidden chars:   ',', ';', '=', '#', '"', "'", '\n', '\r', '\t', ' '
                       -> replaced with '_'
  - Unknown values:    literal string 'unknown'

This module is the single source of truth for how both visualize_benchmarks.py
and compare_benchmarks.py read results. Keep it dependency-light (pandas only).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


# --------------------------------------------------------------------------- #
# Metadata fields promoted to DataFrame columns if not already present.       #
# Kept small on purpose: only what is useful for grouping and for plot text.  #
# --------------------------------------------------------------------------- #
PROMOTE_FIELDS = (
    "host", "cpu", "cores_logical", "cores_physical",
    "compiler", "compiler_version", "compiler_flags", "openmp",
    "run_date", "language", "project", "project_version", "git_commit",
    "iterations", "warmup",
)

_TRUE_STRS = {"true", "True", "TRUE", "1", "yes"}
_FALSE_STRS = {"false", "False", "FALSE", "0", "no"}


def parse_metadata_block(path: str | Path) -> Dict[str, str]:
    """Parse the leading '#' comment lines of a benchmark CSV.

    Returns a dict of key -> value. Stops at the first non-comment line.
    Silently tolerates malformed pairs.
    """
    meta: Dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n\r")
                if not line.startswith("#"):
                    break
                body = line.lstrip("#").strip()
                if not body:
                    continue
                for pair in body.split(";"):
                    pair = pair.strip()
                    if "=" not in pair:
                        continue
                    k, _, v = pair.partition("=")
                    k = k.strip()
                    v = v.strip()
                    if k:
                        meta[k] = v
    except OSError:
        pass
    return meta


def _coerce_is_parallel(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the string 'true'/'false' values to real booleans."""
    if "is_parallel" in df.columns and df["is_parallel"].dtype == object:
        s = df["is_parallel"].astype(str).str.strip()
        df["is_parallel"] = s.isin(_TRUE_STRS)
    return df


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        "min(ms)": "min_ms",
        "median(ms)": "median_ms",
        "mean(ms)": "mean_ms",
        "std(ms)": "std_ms",
        "gflop/s": "gflops",
        "eff(%)": "efficiency_pct",
        "efficiency": "efficiency_pct",
    }
    df.columns = [col_map.get(c.lower(), c.lower().replace(" ", "_")) for c in df.columns]
    return df


def _infer_from_filename(stem: str, df: pd.DataFrame) -> pd.DataFrame:
    if "benchmark" not in df.columns or df["benchmark"].isna().all():
        m = re.match(
            r"(?:scaling_)?(\d*mm|cholesky|correlation|jacobi2d|nussinov|gemm|syrk|heat3d)",
            stem.lower(),
        )
        if m:
            df["benchmark"] = m.group(1)
    if "dataset" not in df.columns or df["dataset"].isna().all():
        for size in ("EXTRALARGE", "LARGE", "MEDIUM", "SMALL", "MINI"):
            if size in stem.upper():
                df["dataset"] = size
                break
    return df


def load_csv(path: str | Path) -> Optional[pd.DataFrame]:
    """Load one benchmark CSV. Metadata is attached at df.attrs['meta'] and
    also promoted to columns listed in PROMOTE_FIELDS."""
    p = Path(path)
    try:
        meta = parse_metadata_block(p)
        # comment='#' is the fix: pandas skips every line beginning with '#'
        df = pd.read_csv(p, comment="#", skip_blank_lines=True)
    except Exception as e:
        print(f"ERROR: cannot read {p.name}: {e}")
        return None

    if df is None or df.empty:
        print(f"WARN: {p.name} has no data rows")
        return df

    df = _normalize_columns(df)
    df = _coerce_is_parallel(df)

    for k in PROMOTE_FIELDS:
        if k in meta and k not in df.columns:
            df[k] = meta[k]

    df = _infer_from_filename(p.stem, df)

    df["source_file"] = p.name
    df.attrs["meta"] = meta
    df.attrs["source_file"] = p.name
    return df


def load_many(paths: Iterable[str | Path]) -> Optional[pd.DataFrame]:
    """Load + concat multiple CSVs."""
    frames: List[pd.DataFrame] = []
    metas: List[Dict[str, str]] = []
    for p in paths:
        df = load_csv(p)
        if df is not None and not df.empty:
            frames.append(df)
            metas.append(df.attrs.get("meta", {}))
    if not frames:
        return None
    out = pd.concat(frames, ignore_index=True)

    merged: Dict[str, str] = {}
    for m in metas:
        for k, v in m.items():
            if v and not merged.get(k):
                merged[k] = v
    out.attrs["meta"] = merged
    return out


# --------------------------------------------------------------------------- #
# Helpers for plot titles / subtitles / footers                               #
# --------------------------------------------------------------------------- #

def run_header_line(meta: Dict[str, str], df: Optional[pd.DataFrame] = None) -> str:
    """Compact run identification suitable for plot subtitles.
    Example: host=cn42 | cpu=Intel_Xeon_E5-2630v3 | cores=16 | cc=gcc-13.2.0
    """
    parts = []
    if meta.get("host"):
        parts.append(f"host={meta['host']}")
    if meta.get("cpu"):
        parts.append(f"cpu={meta['cpu']}")
    cores = meta.get("cores_logical") or meta.get("cores_physical")
    if cores:
        parts.append(f"cores={cores}")
    if meta.get("compiler"):
        cc = meta["compiler"]
        if meta.get("compiler_version"):
            cc = f"{cc}-{meta['compiler_version']}"
        parts.append(f"cc={cc}")
    return " | ".join(parts)


def run_footer_line(meta: Dict[str, str]) -> str:
    """Second-line detail for plot footers: date, iterations, OMP config, git."""
    parts = []
    if meta.get("run_date"):
        parts.append(meta["run_date"])
    its = meta.get("iterations")
    wu = meta.get("warmup")
    if its or wu:
        parts.append(f"iters={its or '?'} (warmup={wu or '?'})")
    for k in ("omp_proc_bind", "omp_places", "omp_schedule"):
        if meta.get(k):
            parts.append(f"{k}={meta[k]}")
    if meta.get("compiler_flags"):
        parts.append(f"flags={meta['compiler_flags']}")
    if meta.get("git_commit"):
        parts.append(f"git={meta['git_commit'][:7]}")
    return " | ".join(parts)


def run_title_suffix(df: pd.DataFrame) -> str:
    """Data-derived subtitle parts: benchmark/dataset/threads."""
    parts = []
    if "benchmark" in df.columns:
        bs = sorted(set(df["benchmark"].dropna().astype(str)))
        if bs:
            parts.append(f"benchmark={','.join(bs)}")
    if "dataset" in df.columns:
        ds = sorted(set(df["dataset"].dropna().astype(str)))
        if ds:
            parts.append(f"dataset={','.join(ds)}")
    if "threads" in df.columns:
        tc = sorted(set(df["threads"].dropna()))
        if tc:
            if len(tc) == 1:
                parts.append(f"threads={int(tc[0])}")
            else:
                parts.append(f"threads={int(min(tc))}..{int(max(tc))}")
    return " | ".join(parts)
