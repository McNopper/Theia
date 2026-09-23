"""clang-tidy over the Ninja compilation database (the classified analysis lane).

The canonical .clang-tidy profile has been config-only; this driver is the
execution lane. It runs clang-tidy (VS LLVM component, resolved via vswhere)
on every project translation unit in build/compile_commands.json in parallel
and classifies findings by the config's WarningsAsErrors contract:

  error categories   clang-diagnostic-*, clang-analyzer-*, bugprone-*  -> FAIL
  report-only        modernize-*, performance-*, portability-*         -> summary

FetchContent/_deps TUs (under the binary dir) are skipped; third-party headers
are already excluded by the config's HeaderFilterRegex. Full per-TU output lands
in build/clang_tidy.log; the console prints the failing findings plus a
per-check summary.

Exit 0 = clean, 1 = error-category findings, 77 = toolchain/DB missing
(ctest SKIP). Run:  python tools/check_tidy.py [--db PATH] [--jobs N]
"""

import concurrent.futures
import json
import os
import re
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VSWHERE = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
TIDY_FALLBACK = r"C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\Llvm\x64\bin\clang-tidy.exe"
DEFAULT_DB = os.path.join(REPO, "build", "compile_commands.json")
DEFAULT_LOG = os.path.join(REPO, "build", "clang_tidy.log")

ERROR_PREFIXES = ("clang-diagnostic-", "clang-analyzer-", "bugprone-")
DIAG = re.compile(r"^(.+):(\d+):(\d+): (?:warning|error): (.+) \[([a-zA-Z0-9_.-]+)\]$",
                  re.MULTILINE)


def find_clang_tidy():
    """Resolve clang-tidy from the newest VS install (vswhere), so a Visual
    Studio upgrade or edition change needs no repo edit."""
    try:
        root = subprocess.check_output(
            [VSWHERE, "-latest", "-property", "installationPath"], text=True
        ).strip()
        p = os.path.join(root, r"VC\Tools\Llvm\x64\bin\clang-tidy.exe")
        if os.path.isfile(p):
            return p
    except (OSError, subprocess.CalledProcessError):
        pass
    return TIDY_FALLBACK


def is_project_tu(path):
    p = path.replace("\\", "/").lower()
    if "/third_party/" in p or "/build/" in p or "/_deps/" in p:
        return False
    return "/src/" in p or "/tests/" in p or "/include/" in p


def sanitize_db(db_path, tmp_dir):
    # Strip C++ module scanning flags that reference files existing only
    # after a build, so the run works on a fresh tree; clang-tidy gets the
    # cl-style commands via cl driver mode (the family compiles clang-cl).
    entries = json.load(open(db_path))
    out = []
    for e in entries:
        if not is_project_tu(e["file"]):
            continue
        parts = e["command"].split()
        parts = [p for p in parts
                 if not p.startswith("@CMakeFiles")
                 and p not in ("-fmodules-ts", "-MD")
                 and not p.startswith("-fmodule-mapper")
                 and not p.startswith("-fdeps-format")]
        e = dict(e)
        e["command"] = " ".join(parts)
        out.append(e)
    db = os.path.join(tmp_dir, "compile_commands.json")
    with open(db, "w") as f:
        json.dump(out, f)
    return out


def run_tu(tidy, tmp_dir, path):
    r = subprocess.run([tidy, "--quiet", "-p", tmp_dir, path,
                        "--extra-arg=--driver-mode=cl"],
                       capture_output=True, text=True)
    return path, r.stdout + r.stderr


def main():
    db = DEFAULT_DB
    jobs = max(4, (os.cpu_count() or 4) - 2)
    args = sys.argv[1:]
    while args:
        a = args.pop(0)
        if a == "--db" and args:
            db = args.pop(0)
        elif a == "--jobs" and args:
            jobs = int(args.pop(0))
    tidy = find_clang_tidy()
    if not os.path.isfile(tidy):
        print("clang-tidy not found: " + tidy)
        return 77
    if not os.path.isfile(db):
        print("compile_commands.json not found: " + db + " (configure build/ first)")
        return 77
    with tempfile.TemporaryDirectory(prefix="check_tidy_") as tmp_dir:
        entries = sanitize_db(db, tmp_dir)
        files = [e["file"] for e in entries]
        outputs = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            futs = {pool.submit(run_tu, tidy, tmp_dir, f): f for f in files}
            done = 0
            for fut in concurrent.futures.as_completed(futs):
                path, text = fut.result()
                outputs[path] = text
                done += 1
                if done % 10 == 0 or done == len(files):
                    print(f"  ... {done}/{len(files)} TUs", flush=True)
    log_path = os.path.join(os.path.dirname(db), "clang_tidy.log") \
        if os.path.dirname(db) else DEFAULT_LOG
    with open(log_path, "w") as f:
        for path in sorted(outputs):
            f.write(f"===== {path}\n{outputs[path]}\n")
    errors = []
    warn_counts = {}
    for path, text in outputs.items():
        for m in DIAG.finditer(text):
            file, line, col, msg, check = m.groups()
            if check.startswith(ERROR_PREFIXES):
                errors.append(f"{file}:{line}:{col}: {msg} [{check}]")
            else:
                warn_counts[check] = warn_counts.get(check, 0) + 1
    if warn_counts:
        top = sorted(warn_counts.items(), key=lambda kv: -kv[1])[:10]
        print("report-only findings (not failing):")
        for check, n in top:
            print(f"  {n:4d}  {check}")
    if errors:
        print(f"\nERROR-CATEGORY FINDINGS ({len(errors)}):")
        for e in errors:
            print("  " + e)
        print(f"\nfull log: {log_path}")
        return 1
    print(f"clang-tidy clean ({len(files)} translation units scanned; log: {log_path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
