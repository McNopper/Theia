#!/usr/bin/env python3
"""check_dead - the generic dead-code gate.

Three evidence classes, each proving something different (and none proving
correctness by itself - every finding needs a human triage before deletion):

1. orphan-headers   - headers nobody #includes (no incoming edge in the
                      measured include graph over the derived scan roots).
2. never-compiled   - .cpp files that are neither in the newest
                      compile_commands.json under build*/ NOR referenced by
                      name in any CMake file (platform-conditional sources
                      are therefore not flagged; "listed but not compiled in
                      this configure" is informational only).
3. fetchcontent     - FetchContent_Declare() names never consumed by any
                      FetchContent_MakeAvailable()/FetchContent_Populate()
                      (name comparison normalized: lowercase, '-' == '_').

Family mode (--family DIR): headers with zero intra-repo consumers get a
second chance against a lazily-built include index of the sibling repos
under DIR - a header orphaned in its own repo may be consumed cross-repo
(the Harmonia core/Env.hpp incident: deleted on a single-repo verdict,
broke Theia's CI). Rescued headers are reported with their consumers; they
are findings no more.

Generic by design: the repo layout is derived (top-level source-containing
directories, like check_arch's detect_roots); nothing is per-repo hardcoded.
Justified findings go into tools/check_dead.exceptions - one entry per line:

    <check> <glob-or-name> <reason>

where <check> is orphan-headers | never-compiled | fetchcontent. Lines
starting with '#' and blank lines are ignored.  Missing exceptions file is
normal.  The file is parsed BOM-tolerantly (utf-8-sig) - Windows tooling
loves to add one.

Exit codes: 0 = clean, 1 = findings, 77 = nothing scanned (skip, per the
analysis-lane SKIP_RETURN_CODE convention).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

CHECK_DEAD_VERSION = "1.1.2"

SKIP_EXIT = 77

# Top-level/vendor names never scanned (mirrors check_arch's VENDOR_DIRS idea).
DEFAULT_EXCLUDES = {
    "build", "_deps", ".git", ".github", ".vs", ".vscode", "node_modules",
    "vendored", "extern", "external", "out", "dist", "cmake-build-debug",
    "cmake-build-release", "coverage", "third_party", "thirdparty",
}

# Excluded directory-name substrings (applied at every depth).
DEFAULT_EXCLUDE_SUBSTRINGS = ("_deps", "CMakeFiles", "__pycache__", "vcpkg_installed")

HEADER_EXTS = {".hpp", ".h", ".hh", ".hxx", ".inl"}
SOURCE_EXTS = {".cpp", ".cc", ".cxx"}

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^">]+)[>"]', re.MULTILINE)
DECLARE_RE = re.compile(r"FetchContent_Declare\s*\(\s*([A-Za-z0-9_\-.]+)")
MAKEAVAIL_RE = re.compile(r"FetchContent_MakeAvailable\s*\(([^)]*)\)")
POPULATE_RE = re.compile(r"FetchContent_Populate\s*\(\s*([A-Za-z0-9_\-.]+)")


def norm(path: str) -> str:
    """Repo-relative, forward-slash, lowercase-normalized path key."""
    return path.replace("\\", "/").lower()


def normalize_dep_name(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def detect_roots(repo: str) -> list[str]:
    """Default scan roots: every top-level directory that contains C/C++
    sources - except vendor/non-source names (mirrors check_arch)."""
    roots = []
    for name in sorted(os.listdir(repo)):
        p = os.path.join(repo, name)
        if (not os.path.isdir(p) or name.startswith(".") or name.startswith("build")
                or name in DEFAULT_EXCLUDES
                or any(x in name for x in DEFAULT_EXCLUDE_SUBSTRINGS)):
            continue
        for dp, dn, fn in os.walk(p):
            dn[:] = [d for d in dn if d not in DEFAULT_EXCLUDES
                     and not any(x in d for x in DEFAULT_EXCLUDE_SUBSTRINGS)]
            exts = {os.path.splitext(f)[1] for f in fn}
            if exts & (HEADER_EXTS | SOURCE_EXTS):
                roots.append(name)
                break
    return roots


def scan_files(repo: str, roots: list[str]) -> list[str]:
    files = []
    for root in roots:
        base = os.path.join(repo, root)
        for dp, dn, fn in os.walk(base):
            dn[:] = [d for d in dn if d not in DEFAULT_EXCLUDES
                     and not any(x in d for x in DEFAULT_EXCLUDE_SUBSTRINGS)]
            for f in fn:
                ext = os.path.splitext(f)[1]
                if ext in HEADER_EXTS or ext in SOURCE_EXTS:
                    rel = os.path.relpath(os.path.join(dp, f), repo)
                    files.append(norm(rel))
    return sorted(set(files))


def load_exceptions(repo: str) -> list[tuple[str, str]]:
    path = os.path.join(repo, "tools", "check_dead.exceptions")
    entries: list[tuple[str, str]] = []
    if not os.path.isfile(path):
        return entries
    with open(path, encoding="utf-8-sig") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 2)
            if len(parts) >= 2:
                entries.append((parts[0], norm(parts[1])))
    return entries


def excepted(check: str, key: str, entries: list[tuple[str, str]]) -> bool:
    for c, pattern in entries:
        if c != check:
            continue
        if pattern in key or key in pattern:  # path-suffix or name match
            return True
    return False


def build_include_edges(repo: str, files: list[str]) -> dict[str, set[str]]:
    """Map every scanned header to the set of files that include it (by any
    spelling that resolves to it: exact relative, or path-suffix match)."""
    by_suffix: dict[str, list[str]] = {}
    for f in files:
        # index each file under progressively shorter suffixes of its path
        parts = f.split("/")
        for i in range(len(parts)):
            by_suffix.setdefault("/".join(parts[i:]), []).append(f)

    edges: dict[str, set[str]] = {f: set() for f in files}
    for f in files:
        try:
            with open(os.path.join(repo, f), encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        for inc in INCLUDE_RE.findall(text):
            key = norm(inc)
            if key in by_suffix:
                for target in by_suffix[key]:
                    if target != f:
                        edges[target].add(f)
    return edges


def find_compile_db(repo: str) -> str | None:
    candidates: list[tuple[float, str]] = []
    for dp, dn, fn in os.walk(repo):
        dn[:] = [d for d in dn if d not in {"_deps", ".git"}]
        if "compile_commands.json" in fn:
            full = os.path.join(dp, "compile_commands.json")
            candidates.append((os.path.getmtime(full), full))
    return max(candidates)[1] if candidates else None


def db_sources(db_path: str) -> set[str]:
    with open(db_path, encoding="utf-8") as fh:
        entries = json.load(fh)
    return {norm(e.get("file", "")) for e in entries}


def cmake_text(repo: str) -> str:
    chunks = []
    for dp, dn, fn in os.walk(repo):
        dn[:] = [d for d in dn if d not in DEFAULT_EXCLUDES
                 and not any(x in d for x in DEFAULT_EXCLUDE_SUBSTRINGS)]
        for f in fn:
            if f == "CMakeLists.txt" or f.endswith(".cmake"):
                try:
                    with open(os.path.join(dp, f), encoding="utf-8",
                              errors="replace") as fh:
                        chunks.append(fh.read())
                except OSError:
                    continue
    return "\n".join(chunks)


def cmake_mentions(repo: str, rel_source: str) -> bool:
    """True when the file NAME (not full path - CMake lists are usually
    relative or bare) appears in any CMake file under the repo."""
    name = os.path.basename(rel_source)
    for dp, dn, fn in os.walk(repo):
        dn[:] = [d for d in dn if d not in DEFAULT_EXCLUDES
                 and not any(x in d for x in DEFAULT_EXCLUDE_SUBSTRINGS)]
        for f in fn:
            if f == "CMakeLists.txt" or f.endswith(".cmake"):
                try:
                    with open(os.path.join(dp, f), encoding="utf-8",
                              errors="replace") as fh:
                        if name in fh.read():
                            return True
                except OSError:
                    continue
    return False


def build_family_includes(family: str, own_repo: str) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """(include-spec -> sibling repos, sibling -> normalized file paths).
    Only called lazily when orphan candidates exist."""
    index: dict[str, set[str]] = {}
    member_files: dict[str, set[str]] = {}
    if not os.path.isdir(family):
        return index, member_files
    for name in sorted(os.listdir(family)):
        member = os.path.join(family, name)
        if (not os.path.isdir(member) or os.path.abspath(member) == os.path.abspath(own_repo)
                or name.startswith(".") or name in DEFAULT_EXCLUDES
                or any(x in name for x in DEFAULT_EXCLUDE_SUBSTRINGS)):
            continue
        for dp, dn, fn in os.walk(member):
            dn[:] = [d for d in dn if d not in DEFAULT_EXCLUDES
                     and not any(x in d for x in DEFAULT_EXCLUDE_SUBSTRINGS)]
            for f in fn:
                ext = os.path.splitext(f)[1]
                if ext not in HEADER_EXTS and ext not in SOURCE_EXTS:
                    continue
                rel = norm(os.path.relpath(os.path.join(dp, f), member))
                member_files.setdefault(name, set()).add(rel)
                try:
                    with open(os.path.join(dp, f), encoding="utf-8",
                              errors="replace") as fh:
                        text = fh.read()
                except OSError:
                    continue
                for inc in INCLUDE_RE.findall(text):
                    index.setdefault(norm(inc), set()).add(name)
    return index, member_files


def family_consumes(index: dict[str, set[str]], member_files: dict[str, set[str]],
                    key: str) -> set[str]:
    """Sibling repos whose include specs resolve to the header `key`
    (exact, or suffix match in either direction - the same philosophy as
    the intra-repo resolution).  A sibling that contains its own COPY of
    the header (same path modulo leading components - e.g. a vendored
    snapshot of the family) is self-referential, not a consumer, and is
    excluded."""
    consumers: set[str] = set()
    variants = {key}
    parts = key.split("/")
    if len(parts) > 1:
        variants.add("/".join(parts[1:]))
    for spec, repos in index.items():
        if spec == key or key.endswith("/" + spec) or spec.endswith("/" + key):
            for repo in repos:
                files = member_files.get(repo, set())
                copied = any(v in files or any(f.endswith("/" + v) for f in files)
                             for v in variants)
                if not copied:
                    consumers.add(repo)
    return consumers


def run(repo: str, report: bool, family: str | None) -> int:
    roots = detect_roots(repo)
    if not roots:
        print("check_dead: no source roots found - nothing to scan")
        return SKIP_EXIT
    files = scan_files(repo, roots)
    headers = [f for f in files if os.path.splitext(f)[1] in HEADER_EXTS]
    sources = [f for f in files if os.path.splitext(f)[1] in SOURCE_EXTS]
    exceptions = load_exceptions(repo)

    if report:
        print(f"roots: {', '.join(sorted(roots))}")
        print(f"scanned: {len(files)} files ({len(headers)} headers, {len(sources)} sources)")

    findings: dict[str, list[str]] = {"orphan-headers": [], "never-compiled": [],
                                      "fetchcontent": []}
    rescued: list[tuple[str, set[str]]] = []

    # 1. orphan headers - zero incoming include edges (intra-repo; family
    #    mode gives cross-repo consumers a second look)
    edges = build_include_edges(repo, files)
    orphan_candidates = [h for h in headers if not edges[h]]
    family_index: dict[str, set[str]] | None = None
    if orphan_candidates and family:
        family_index, member_files = build_family_includes(family, repo)
    for h in orphan_candidates:
        if excepted("orphan-headers", h, exceptions):
            continue
        if family_index is not None:
            consumers = family_consumes(family_index, member_files, h)
            if consumers:
                rescued.append((h, consumers))
                continue
        findings["orphan-headers"].append(h)

    # 2. never-compiled - not in compile db AND not mentioned in any CMake file
    db_path = find_compile_db(repo)
    compiled: set[str] = set()
    if db_path:
        compiled = db_sources(db_path)
    listed_but_not_compiled = 0
    for s in sources:
        in_db = s in compiled or norm(os.path.join(repo, s)) in compiled
        if in_db:
            continue
        if cmake_mentions(repo, s):
            listed_but_not_compiled += 1  # informational (e.g. other platform)
            continue
        if not excepted("never-compiled", s, exceptions):
            findings["never-compiled"].append(s)
    if db_path and report:
        print(f"compile db: {os.path.relpath(db_path, repo)} "
              f"({len(compiled)} TUs; {listed_but_not_compiled} listed-not-compiled-here)")

    # 3. FetchContent declare/consume mismatch
    text = cmake_text(repo)
    declared = {normalize_dep_name(m) for m in DECLARE_RE.findall(text)}
    consumed = {normalize_dep_name(n) for group in MAKEAVAIL_RE.findall(text)
                for n in group.split()}
    consumed |= {normalize_dep_name(m) for m in POPULATE_RE.findall(text)}
    if any("$" in a for group in MAKEAVAIL_RE.findall(text) for a in group.split()):
        print("fetchcontent: MakeAvailable uses variable lists - check skipped")
    else:
        for name in sorted(declared - consumed):
            if not excepted("fetchcontent", name, exceptions):
                findings["fetchcontent"].append(name)

    # Report
    total = sum(len(v) for v in findings.values())
    for h, consumers in sorted(rescued):
        print(f"info  cross-repo consumer: {h} (consumed by {', '.join(sorted(consumers))})")
    for check, items in findings.items():
        label = "PASS" if not items else "FAIL"
        print(f"{label} {check}: {len(items)}")
        for item in items:
            print(f"    {item}")
    if total == 0:
        print("check_dead: clean")
        return 0
    print(f"check_dead: {total} finding(s) - triage before deleting anything; "
          f"justified cases go into tools/check_dead.exceptions")
    return 1


def main(argv=None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        prog="check_dead.py",
        description="the generic dead-code gate (orphan headers, never-compiled "
                    "sources, FetchContent declare/consume mismatches)")
    ap.add_argument("--version", action="version", version=f"check_dead {CHECK_DEAD_VERSION}")
    ap.add_argument("--repo", default=os.path.dirname(here),
                    help="repository root to check (default: the directory above this script)")
    ap.add_argument("--family", default=None,
                    help="directory containing sibling repos - orphan-header candidates "
                         "get a second look against their include graphs (cross-repo "
                         "consumers rescue the header)")
    ap.add_argument("--report", action="store_true",
                    help="also print the derived scan roots and file counts")
    args = ap.parse_args(argv)
    return run(os.path.abspath(args.repo), args.report, args.family)


if __name__ == "__main__":
    sys.exit(main())
