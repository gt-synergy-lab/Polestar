#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MBPP lm-eval samples(.jsonl) -> cleaned completions + pass@1 export.

Expected per-line JSON shape (like your samples_mbpp_*.jsonl):
- doc: { task_id, test_list, test_setup_code, ... }
- resps: [[<model_text>], ...]  (we use first sample)
- pass_at_1: 0/1  (already computed by your eval run)

Usage:
  python eval_mbpp_samples.py samples_mbpp.jsonl
  python eval_mbpp_samples.py samples_mbpp.jsonl --out samples_mbpp.jsonl.cleaned
  python eval_mbpp_samples.py samples_mbpp.jsonl --run-tests --timeout 3
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import multiprocessing as mp
import os
import re
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def write_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


_CODEBLOCK_PY = re.compile(r"```python\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_CODEBLOCK_ANY = re.compile(r"```(?:\w+)?\s*(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """
    Extract code from model output.
    Priority:
      1) ```python ... ```
      2) ``` ... ```
      3) whole text
    Also strips trailing [DONE].
    """
    if text is None:
        return ""
    s = text.replace("[DONE]", "").strip()

    m = _CODEBLOCK_PY.search(s)
    if m:
        return m.group(1).strip()

    m = _CODEBLOCK_ANY.search(s)
    if m:
        return m.group(1).strip()

    return s


def strip_embedded_tests(code: str, tests: List[str]) -> str:
    """
    If the model pasted the tests into its answer, remove them.
    We try:
      - truncate before the first exact test string occurrence
      - otherwise truncate before the first top-level 'assert ' line
    """
    if not code:
        return code

    # 1) exact test occurrence
    idxs = []
    for t in tests or []:
        pos = code.find(t)
        if pos != -1:
            idxs.append(pos)
    if idxs:
        code = code[: min(idxs)].rstrip()

    # 2) first assert line (best-effort)
    lines = code.splitlines()
    cut = None
    for i, line in enumerate(lines):
        if line.lstrip().startswith("assert "):
            cut = i
            break
        if line.strip() in ("[BEGIN]", "[DONE]"):
            cut = i
            break
    if cut is not None and cut > 0:
        code = "\n".join(lines[:cut]).rstrip()

    return code


def _exec_worker(program: str, q: mp.Queue) -> None:
    buf = io.StringIO()
    try:
        g: Dict[str, Any] = {}
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(program, g, g)
        q.put(
            {
                "ok": True,
                "output": buf.getvalue(),
                "exc_type": None,
                "exc_msg": None,
                "traceback": None,
            }
        )
    except Exception as e:
        q.put(
            {
                "ok": False,
                "output": buf.getvalue(),
                "exc_type": type(e).__name__,
                "exc_msg": str(e),
                "traceback": traceback.format_exc(),
            }
        )


def run_with_timeout(program: str, timeout_s: float) -> Dict[str, Any]:
    q: mp.Queue = mp.Queue()
    p = mp.Process(target=_exec_worker, args=(program, q))
    p.start()
    p.join(timeout_s)

    if p.is_alive():
        p.terminate()
        p.join(1.0)
        return {
            "ok": False,
            "output": "",
            "exc_type": "TimeoutError",
            "exc_msg": f"Timed out after {timeout_s}s",
            "traceback": None,
        }

    if q.empty():
        return {
            "ok": False,
            "output": "",
            "exc_type": "RuntimeError",
            "exc_msg": "No result returned from worker",
            "traceback": None,
        }

    return q.get()


def eval_one_by_running_tests(sample: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    doc = sample.get("doc", {}) or {}
    tests: List[str] = doc.get("test_list") or []
    setup: str = doc.get("test_setup_code") or ""

    # model output text (first candidate)
    resps = sample.get("resps") or sample.get("filtered_resps")
    if isinstance(resps, list) and resps and isinstance(resps[0], list) and resps[0]:
        model_text = resps[0][0]
    elif isinstance(resps, list) and resps:
        model_text = resps[0]
    else:
        model_text = ""

    code = strip_embedded_tests(extract_code(model_text), tests)
    program = "\n".join([setup, code, *tests]).strip() + "\n"

    r = run_with_timeout(program, timeout_s=timeout_s)
    return {
        "completion": code,
        "ran_tests": True,
        "passed": bool(r["ok"]),
        "error_type": r["exc_type"],
        "error_msg": r["exc_msg"],
        "traceback": r["traceback"],
        "output": r["output"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("samples_jsonl", help="lm-eval samples jsonl (e.g., samples_mbpp_*.jsonl)")
    ap.add_argument("--out", default=None, help="output cleaned jsonl path (default: <in>.cleaned.jsonl)")
    ap.add_argument("--run-tests", action="store_true", help="re-run doc.test_list asserts in a subprocess")
    ap.add_argument("--timeout", type=float, default=3.0, help="timeout seconds per sample when --run-tests")
    args = ap.parse_args()

    inp = args.samples_jsonl
    out = args.out or (inp + ".cleaned.jsonl")

    data = read_jsonl(inp)

    rows: List[Dict[str, Any]] = []
    pass_list: List[float] = []

    for s in data:
        doc = s.get("doc", {}) or {}
        task_id = doc.get("task_id", s.get("doc_id"))

        # Always extract completion
        resps = s.get("resps") or s.get("filtered_resps")
        if isinstance(resps, list) and resps and isinstance(resps[0], list) and resps[0]:
            model_text = resps[0][0]
        elif isinstance(resps, list) and resps:
            model_text = resps[0]
        else:
            model_text = ""

        tests = doc.get("test_list") or []
        completion = strip_embedded_tests(extract_code(model_text), tests)

        row: Dict[str, Any] = {
            "task_id": task_id,
            "doc_id": s.get("doc_id"),
            "completion": completion,
        }

        if args.run_tests:
            r = eval_one_by_running_tests(s, timeout_s=args.timeout)
            row.update(
                {
                    "passed": r["passed"],
                    "error_type": r["error_type"],
                    "error_msg": r["error_msg"],
                }
            )
            # When we re-run tests, pass@1 is 1/0 (single completion)
            row["pass_at_1"] = 1.0 if r["passed"] else 0.0
            pass_list.append(row["pass_at_1"])
        else:
            # Use existing pass_at_1 if present (your file has it)
            if "pass_at_1" in s:
                row["pass_at_1"] = float(s["pass_at_1"])
                pass_list.append(row["pass_at_1"])
            # Optional: store raw for debugging
            # row["raw_response"] = model_text

        rows.append(row)

    write_jsonl(rows, out)

    if pass_list:
        avg = sum(pass_list) / len(pass_list)
        print(f"pass@1 = {avg:.6f}  ({len(pass_list)} samples)")
    else:
        print(f"Wrote {len(rows)} rows to {out} (no pass_at_1 available)")

    print(f"Saved: {out}")


if __name__ == "__main__":
    # safer start method on some systems
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass
    main()
