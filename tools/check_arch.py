#!/usr/bin/env python3
"""check_arch.py - the generic architecture gate: analyze, define, check.

DEVELOPMENT COPY - lives OUTSIDE every repository until placement is decided.

THE IDEA (reflexion models, Murphy/Notkin/Sullivan 1995, applied as an
architecture fitness function): the high-level model is POSITED in a
markdown document, the source model is EXTRACTED from the include graph,
and the check reflexes one against the other. The document IS the
architecture contract.

SEPARATION OF CONCERNS (one document, two kinds of content):
  - THE RULES  live in the `arch` fenced block - the machine contract:
    knobs (scan/resolve/gates), `layer` declarations (the STACK),
    `module` declarations (optionally `in` a layer) and `allow A -> B`
    edges. DEFAULT DENY: an edge that is not declared is a violation -
    a module is distinct because its dependencies are declared.
  - THE DIAGRAMS (block-beta) are pure VISUALIZATION: they are GENERATED
    from the rules (`--diagram`) into the marked section and drift-gated
    on every check - never hand-kept, never parsed. THE DRILL-DOWN: the
    top diagram resolves the layer stack (each block IS a layer), then
    every layer resolves into its modules, then every container module
    into its direct children - level by level until the final module
    level, where this tool stops (a class diagram makes more sense
    there). A block spans by its PULL (1 + the declared edges leaving
    its subtree): the more a block draws from the rest of the
    architecture, the wider it renders; `space` fills the slack.

LAYERS (the stack):
  `layer core` declares a layer; the DECLARATION ORDER is the stack,
  bottom-up (the first layer depends on nothing above it). A module
  joins a layer with `module theia/scene in theia`. The stack is LAW:
  a declared edge that runs from a higher layer DOWN-REFERENCING is the
  only legal direction - an edge from layer X to a layer ABOVE X is a
  violation even though it is declared. Modules without a layer are
  outside the stack (unconstrained by it).

THE THREE-STEP MODEL:
  1. ANALYZE  `--derive` scans the sources, builds the measured include
     graph and emits a draft ARCHITECTURE.md: the modules AS THEY ARE,
     the layers AS THEY MEASURE, the dependencies AS THEY ARE (cycles
     reported for fixing first).
  2. DEFINE   the reviewed (and tightened) document becomes the checked-in
     contract: "this is the architecture".
  3. CHECK    the default mode proves the code against that contract on
     every run: DIRECTION (only DECLARED edges), STACK (layer order),
     CLASSIFICATION (every file maps to a declared module), CYCLES
     (file-level SCC), DIAGRAM DRIFT, and - opt-in - NAMESPACES.

THE `arch` BLOCK (the whole contract; 100% generic - the engine knows no
repo, no layer name, no namespace):

  ```arch
  scan: src                      # comma-separated roots (repo-relative)
  extensions: .hpp, .cpp         # scanned source extensions
  exclude: build, _deps, .git    # substring-excluded directories
  unclassified: error            # error | ignore - files off the module map
  external: harmonia, slang-math # other-repo includes (reported, allowed)
  forbidden:                     # hard violations (e.g. sibling repos)
  quotes_relative: true          # quoted includes resolve includer-relative
  repo_fallback: false           # ... then against the repo root
  cycles: true                   # the file-level cycle gate
  graph: regex                   # regex | scan-deps - the graph source
  db: build/ninja/compile_commands.json   # scan-deps: the compile database
  clang_scan_deps:               # scan-deps: binary (else PATH / vswhere)
  nesting: off                   # off | error - the NESTING LAW: a module
                                 # must not depend on its own child module
                                 # (the deeper module uses its ancestors,
                                 # never the reverse)
  unresolved: info               # info | error - quoted includes that
                                 # resolve to nothing
  namespace_root:                # opt-in namespace gate
  namespace_module:              # the module that IS the root namespace
  namespace_require: .hpp        # extensions that must ALWAYS open it

  layer common                   # the stack, bottom-up:
  layer platform                 # first declared = the bottom layer
  layer core

  module common/assets in common # a module = a path prefix in the tree
  module platform/sdl in platform
  module core/loader in core
  module demo                    # unlayered: outside the stack

  allow core/loader -> common/assets   # an edge = "left may use right"
  ```
  `#` comments and blank lines are free; `allow` edges must reference
  declared modules (a typo is a contract error, never a silent no-op).

THE EXACT GRAPH (`graph: scan-deps`): instead of the regex scan, the
dependency graph comes from clang-scan-deps over the compile database -
the preprocessor truth, conditional includes included. The database is
CONTRACTED: a database whose translation units vanished on disk is
REFUSED (exit 2), and every compiled TU must be scanned or the run fails
(exit 1) - a partial scan never passes silently. In this mode
forbidden/external match against the resolved repo-relative paths, and
`resolve`/`unresolved` do not apply (the graph is exact).

Exit codes: 0 clean - 1 violations - 2 config/usage error - 77 skipped.
Usage: python tools/check_arch.py [--doc FILE] [--report] [--diagram]
                                  [--derive [--scan DIRS] [--out FILE]]
                                  [--selftest] [-h]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from shutil import which

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EXIT_OK = 0
EXIT_VIOLATIONS = 1
EXIT_USAGE = 2
EXIT_SKIP = 77

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]', re.MULTILINE)
ARCH_BLOCK_RE = re.compile(r"```arch[ \t]*\r?\n(.*?)```", re.DOTALL)
DIAGRAM_BEGIN = "<!-- check_arch: diagram BEGIN -->"
DIAGRAM_END = "<!-- check_arch: diagram END -->"
DEFAULT_EXTS = [".hpp", ".cpp", ".h", ".c"]
DEFAULT_EXCLUDES = ["build", "_deps", ".git"]
DOC_CANDIDATES = ["docs/ARCHITECTURE.md", "ARCHITECTURE.md", "tools/ARCHITECTURE.md"]
VSWHERE = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"

KNOB_KEYS = {
    "scan", "extensions", "exclude", "unclassified", "external", "forbidden",
    "quotes_relative", "repo_fallback", "cycles", "graph", "db", "clang_scan_deps",
    "nesting", "unresolved",
    "namespace_root", "namespace_module", "namespace_require",
}


class ContractError(Exception):
    """A contract problem the user must fix (exit 2)."""


def _strip(s):
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        s = s[1:-1].strip()
    return s


def _safe_id(path):
    return re.sub(r"[^A-Za-z0-9_]", "_", path)


class Contract:
    """The parsed architecture contract (the `arch` block of the document)."""

    def __init__(self, text, path):
        self.path = path
        blocks = ARCH_BLOCK_RE.findall(text)
        if not blocks:
            raise ContractError(f"{path}: no ```arch block (the machine contract)")
        knobs = {}
        layers = []
        modules = []   # (path, layer-or-None)
        edges = []
        for block in blocks:
            for raw in block.splitlines():
                line = raw.split("#")[0].strip()
                if not line:
                    continue
                m = re.match(r"^layer\s+(\S+)$", line)
                if m:
                    layers.append(m.group(1))
                    continue
                m = re.match(r"^module\s+(\S+)(?:\s+in\s+(\S+))?$", line)
                if m:
                    modules.append((m.group(1), m.group(2)))
                    continue
                m = re.match(r"^allow\s+(\S+)\s*->\s*(\S+)$", line)
                if m:
                    edges.append((m.group(1), m.group(2)))
                    continue
                if ":" in line:
                    k, v = line.split(":", 1)
                    if k.strip() not in KNOB_KEYS:
                        raise ContractError(f"{path}: unknown knob {k.strip()!r} "
                                            f"(known: {sorted(KNOB_KEYS)})")
                    knobs[k.strip()] = v.strip()
                    continue
                raise ContractError(f"{path}: cannot parse contract line: {raw.strip()!r}")

        dupes = sorted({l for l in layers if layers.count(l) > 1})
        if dupes:
            raise ContractError(f"{path}: duplicate layer declarations {dupes}")
        self.stack = layers                    # bottom-up: first = the bottom
        self.rank = {name: i for i, name in enumerate(layers)}
        dupes = sorted({p for p, _ in modules if [x for x, _ in modules].count(p) > 1})
        if dupes:
            raise ContractError(f"{path}: duplicate module declarations {dupes}")
        self.modules = {p for p, _ in modules}
        self.layer_of = {}
        for p, lay in modules:
            if lay is not None and lay not in self.rank:
                raise ContractError(f"{path}: module {p} in undeclared layer {lay!r} "
                                    f"(declare it with 'layer')")
            if lay is not None:
                self.layer_of[p] = lay
        self.edges = set(edges)
        for a, b in edges:
            missing = [x for x in (a, b) if x not in self.modules]
            if missing:
                raise ContractError(f"{path}: edge {a} -> {b} references undeclared "
                                    f"module(s) {missing} (declare them with 'module')")
        selfloops = sorted({a for a, b in edges if a == b})
        if selfloops:
            raise ContractError(f"{path}: self-edges are implicit, drop them: {selfloops}")

        def lst(key, default):
            v = knobs.get(key)
            return [x.strip() for x in v.split(",") if x.strip()] if v else default

        def flag(key, default):
            v = knobs.get(key)
            if v is None or v == "":
                return default
            if v.lower() in ("true", "yes", "on"):
                return True
            if v.lower() in ("false", "no", "off"):
                return False
            raise ContractError(f"{path}: {key}: true/false expected, got {v!r}")

        self.roots = lst("scan", [])
        if not self.roots:
            raise ContractError(f"{path}: the contract needs 'scan:' (at least one root)")
        self.extensions = lst("extensions", DEFAULT_EXTS)
        self.exclude_dirs = lst("exclude", DEFAULT_EXCLUDES)
        self.unclassified = knobs.get("unclassified") or "error"
        if self.unclassified not in ("error", "ignore"):
            raise ContractError(f"{path}: unclassified: error|ignore expected")
        self.external = lst("external", [])
        self.forbidden = lst("forbidden", [])
        self.relative_quotes = flag("quotes_relative", True)
        self.repo_fallback = flag("repo_fallback", False)
        self.cycles = flag("cycles", True)
        self.ns_root = knobs.get("namespace_root") or None
        self.ns_module = knobs.get("namespace_module") or None
        if bool(self.ns_root) != bool(self.ns_module):
            raise ContractError(f"{path}: namespace_root and namespace_module go together")
        self.ns_require_all = lst("namespace_require", [])

        self.graph = knobs.get("graph") or "regex"
        if self.graph not in ("regex", "scan-deps"):
            raise ContractError(f"{path}: graph: regex|scan-deps expected, got {self.graph!r}")
        self.db = knobs.get("db") or ""
        if self.graph == "scan-deps" and not self.db:
            raise ContractError(f"{path}: graph: scan-deps needs 'db:' (the compile database)")
        self.clang_scan_deps = (knobs.get("clang_scan_deps")
                                or os.environ.get("CHECK_ARCH_CLANG_SCAN_DEPS") or "")
        self.nesting = knobs.get("nesting") or "off"
        if self.nesting not in ("off", "error"):
            raise ContractError(f"{path}: nesting: off|error expected, got {self.nesting!r}")
        self.unresolved_policy = knobs.get("unresolved") or "info"
        if self.unresolved_policy not in ("info", "error"):
            raise ContractError(f"{path}: unresolved: info|error expected")

    def module_of(self, rel):
        """File (posix, scan-root-stripped) -> declared module (longest prefix)."""
        best = None
        for m in self.modules:
            if rel == m or rel.startswith(m + "/"):
                if best is None or len(m) > len(best):
                    best = m
        return best

    def _direct_children(self, m):
        return sorted(c for c in self.modules
                      if c.startswith(m + "/") and "/" not in c[len(m) + 1:])

    def _container_modules(self):
        return sorted(m for m in self.modules if self._direct_children(m))

    def _root_module(self, m):
        parent = m.rsplit("/", 1)[0] if "/" in m else None
        return parent not in self.modules

    def _subtree(self, node):
        return {m for m in self.modules if m == node or m.startswith(node + "/")}

    def _pull_weight(self, node):
        """1 + the declared edges leaving node's subtree - THE PULL: how much
        this node draws from the rest of the architecture."""
        sub = self._subtree(node)
        return 1 + sum(1 for a, b in self.edges if a in sub and b not in sub)

    def _layer_pull(self, lay):
        sub = {m for m in self.modules if self.layer_of.get(m) == lay}
        return 1 + sum(1 for a, b in self.edges if a in sub and b not in sub)

    def _fall_levels(self, group):
        """Dependency levels within a group (leaf = 0; a cycle caps at the
        group size, deterministically). Returns {member: level}."""
        group = set(group)
        deps = {m: {b for (a, b) in self.edges if a == m and b in group and b != m}
                for m in group}
        level = {m: 0 for m in group}
        for _ in range(len(group) + 1):
            changed = False
            for m in sorted(group):
                if deps[m]:
                    want = min(1 + max(level[d] for d in deps[m]), len(group))
                    if want != level[m]:
                        level[m] = want
                        changed = True
            if not changed:
                break
        return level

    def _fall_rows(self, members):
        """The dependency fall of a group, top-down: consumers above, used
        modules below. [(comment, [(member, pull weight)])]."""
        level = self._fall_levels(members)
        by_level = {}
        for m in sorted(members):
            by_level.setdefault(level[m], []).append(m)
        return [(None, [(m, self._pull_weight(m)) for m in by_level[lvl]])
                for lvl in sorted(by_level, reverse=True)]

    def _block_text(self, rows):
        """One block-beta diagram: rows top-down; a block spans EXACTLY its
        pull weight (the more it draws from the rest, the wider), `space`
        fills the row's slack (mermaid block column-width dynamics)."""
        width = max((sum(w for _, w in row) for _, row in rows), default=1)
        out = ["block-beta", f"    columns {width}"]
        for comment, row in rows:
            if comment:
                out.append(f"    %% {comment}")
            for m, w in row:
                out.append(f'    {_safe_id(m)}["{m}"]' + (f":{w}" if w > 1 else ""))
            slack = width - sum(w for _, w in row)
            if slack:
                out.append("    space" if slack == 1 else f"    space:{slack}")
        return "\n".join(out)

    def diagram_sections(self):
        """[(heading, block_text)]: THE DRILL-DOWN. The top diagram resolves
        the layer stack (each block IS a layer, spanned by its pull); then
        every layer resolves into its modules; then every container module
        into its direct children - and so on, level by level, until the
        final module level. A leaf module gets no diagram: this tool stops
        exactly where class diagrams would make more sense."""
        sections = []
        # --- the top: the layer stack (each block = one layer)
        rows = []
        for lay in reversed(self.stack):
            if any(self.layer_of.get(m) == lay for m in self.modules):
                rows.append((None, [(lay, self._layer_pull(lay))]))
        unlayered = sorted(m for m in self.modules if m not in self.layer_of
                           and self._root_module(m))
        if unlayered:
            rows.append(("unlayered", [(m, self._pull_weight(m)) for m in unlayered]))
        if rows:
            sections.append(("the layers", self._block_text(rows)))
        # --- each layer resolves into its (top-level) modules
        for lay in reversed(self.stack):
            members = sorted(m for m in self.modules if self.layer_of.get(m) == lay
                             and self._root_module(m))
            if members:
                sections.append((f"layer {lay} - the modules",
                                 self._block_text(self._fall_rows(members))))
        if any(m not in self.layer_of for m in self.modules):
            unlayered_all = sorted(m for m in self.modules if m not in self.layer_of
                                   and self._root_module(m))
            if unlayered_all:
                sections.append(("unlayered - the modules",
                                 self._block_text(self._fall_rows(unlayered_all))))
        # --- then each container module resolves into its children, and so on
        for m in self._container_modules():
            sections.append((f"{m} - the nested modules",
                             self._block_text(self._fall_rows(self._direct_children(m)))))
        return sections

    def marked_diagram(self):
        """The whole marked section: every generated diagram of the drill-down."""
        return "\n\n".join(f"### {heading}\n\n```mermaid\n{text}\n```"
                           for heading, text in self.diagram_sections())
        rows = []
        for lay in reversed(self.stack):
            members = sorted(m for m in self.modules if self.layer_of.get(m) == lay)
            if members:
                rows.append((lay, members))
        unlayered = sorted(m for m in self.modules if m not in self.layer_of)
        if unlayered:
            rows.append((None, unlayered))
        if not rows:
            return "block-beta"
        below = {m: 0 for m in self.modules}
        for a, b in self.edges:
            la, lb = self.layer_of.get(a), self.layer_of.get(b)
            if la and lb and self.rank[lb] < self.rank[la]:
                below[a] += 1
        weight = {m: 1 + below[m] for m in self.modules}
        width = max(sum(weight[m] for m in members) for _, members in rows)

        def spans(members):
            """Proportional split of the row's own weight sum; the slack
            (width - total) becomes `space`."""
            total = sum(weight[m] for m in members)
            exact = [width * weight[m] / total for m in members]
            base = [int(x) for x in exact]
            order = sorted(range(len(members)),
                           key=lambda i: (-(exact[i] - base[i]), members[i]))
            for i in order[:width - sum(base)]:
                base[i] += 1
            return base

        out = ["block-beta", f"    columns {width}"]
        for lay, members in rows:
            out.append(f"    %% layer: {lay if lay else 'unlayered'}")
            sp = spans(members)
            for m, s in zip(members, sp):
                out.append(f'    {_safe_id(m)}["{m}"]' + (f":{s}" if s > 1 else ""))
            slack = width - sum(sp)
            if slack:
                out.append("    space" if slack == 1 else f"    space:{slack}")
        return "\n".join(out)


def load_contract_text(repo, doc_arg):
    if doc_arg:
        p = doc_arg if os.path.isabs(doc_arg) else os.path.join(repo, doc_arg)
    else:
        p = next((os.path.join(repo, c) for c in DOC_CANDIDATES if os.path.isfile(
            os.path.join(repo, c))), None)
        if p is None:
            raise ContractError("no contract document found (looked for "
                                + ", ".join(DOC_CANDIDATES)
                                + "; run check_arch.py --derive to draft one)")
    if not os.path.isfile(p):
        raise ContractError(f"contract document not found: {p}")
    with open(p, encoding="utf-8") as fh:
        return p, fh.read()


def scan_files(contract, repo):
    files = {}
    for root in contract.roots:
        base = os.path.normpath(os.path.join(repo, root))
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(
                d for d in dirnames
                if not any(x in os.path.join(dirpath, d) for x in contract.exclude_dirs))
            for name in sorted(filenames):
                if os.path.splitext(name)[1] in contract.extensions:
                    p = os.path.normpath(os.path.join(dirpath, name))
                    files[p] = os.path.relpath(p, repo).replace(os.sep, "/")
    return files


def _prefix_bases(contract, repo):
    """Rooted-include bases derived from the scan roots: a first segment that
    is a real directory under a scan root maps to that root
    (`aether/types.hpp` with src/aether/ -> src/)."""
    if not hasattr(contract, "_pbases"):
        pbases = []
        for root in contract.roots:
            base = os.path.join(repo, root)
            if not os.path.isdir(base):
                continue
            for name in sorted(os.listdir(base)):
                if os.path.isdir(os.path.join(base, name)):
                    pbases.append((name + "/", root))
        contract._pbases = sorted(set(pbases), key=lambda x: -len(x[0]))
    return contract._pbases


def resolve(contract, repo, inc, kind, from_file, files):
    for prefix, base in _prefix_bases(contract, repo):
        if inc.startswith(prefix):
            t = os.path.normpath(os.path.join(repo, base, inc))
            return t if t in files else None
    if kind == '"' and contract.relative_quotes:
        t = os.path.normpath(os.path.join(os.path.dirname(from_file), inc))
        if t in files:
            return t
        if contract.repo_fallback:
            t = os.path.normpath(os.path.join(repo, inc))
            if t in files:
                return t
    return None


def build_graph(contract, repo, files):
    adj = {p: set() for p in files}
    file_mod = {}
    for p, rel in files.items():
        stripped = rel
        for root in contract.roots:
            if stripped.startswith(root + "/"):
                stripped = stripped[len(root) + 1:]
                break
        # a module prefix matches with or without the scan-root segment
        # (roots may be technical containers like src/, or architectural
        # units like app/, engine/)
        file_mod[p] = contract.module_of(stripped) or contract.module_of(rel)
    mod_edges = {}
    ext_counts = {}
    forbidden_hits = []
    unresolved = []
    for p, rel in files.items():
        with open(p, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        for m in INCLUDE_RE.finditer(text):
            kind, inc = m.group(1), m.group(2)
            if any(inc.startswith(x) for x in contract.forbidden):
                forbidden_hits.append(f'{rel}: must not include "{inc}" (forbidden dependency)')
                continue
            if any(inc.split("/")[0] == x or inc.startswith(x + "/") for x in contract.external):
                key = inc.split("/")[0]
                ext_counts[key] = ext_counts.get(key, 0) + 1
                continue
            t = resolve(contract, repo, inc, kind, p, files)
            if t is None:
                if kind == '"':
                    unresolved.append(f'{rel}: "{inc}"')
                continue
            if t == p:
                continue
            adj[p].add(t)
            a, b = file_mod[p], file_mod[t]
            if a is not None and b is not None and a != b:
                mod_edges[(a, b)] = mod_edges.get((a, b), 0) + 1
    return adj, file_mod, mod_edges, ext_counts, forbidden_hits, unresolved


def parse_make_deps(text):
    """clang-scan-deps make-style output -> {tu: set(include paths)}.
    (Ported from GameBits' proven parser: $-continued dependency blocks,
    the TU first, headers below.)"""
    deps = {}
    current = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.endswith("\\"):
            line = line[:-1].strip()
        if line.endswith(":"):
            current = None
            continue
        if not (line.startswith("/") or (len(line) > 2 and line[1] == ":")):
            continue
        p = line.replace("\\", "/")
        if current is None and p.lower().endswith((".cpp", ".c", ".cc", ".cxx")):
            current = p
            deps[current] = set()
        elif current is not None and p != current:
            deps[current].add(p)
    return deps


def find_clang_scan_deps(explicit):
    """The scan tool: explicit knob / env override, then PATH, then the
    newest Visual Studio's LLVM component (vswhere) - a VS upgrade or
    edition change needs no repo edit."""
    if explicit:
        return explicit
    p = which("clang-scan-deps")
    if p:
        return p
    try:
        root = subprocess.check_output(
            [VSWHERE, "-latest", "-property", "installationPath"], text=True).strip()
        cand = os.path.join(root, r"VC\Tools\Llvm\x64\bin\clang-scan-deps.exe")
        if os.path.isfile(cand):
            return cand
    except (OSError, subprocess.CalledProcessError):
        pass
    return "clang-scan-deps"


TU_EXTS = (".cpp", ".c", ".cc", ".cxx")
MODULE_SCAN_FLAGS = ("-fmodules-ts", "-MD")


def build_graph_scan_deps(contract, repo, files):
    """The EXACT graph: clang-scan-deps over the compile database.

    -> (graph-6-tuple, error_code, error_lines): on success the tuple and
    (0, []); a refused database (missing, vanished TUs, tool failure) is
    (None, EXIT_USAGE, [...]); an incomplete scan (a compiled TU the scan
    missed) never passes silently: (None, EXIT_VIOLATIONS, [...])."""
    db = os.path.normpath(os.path.join(repo, contract.db))
    if not os.path.isfile(db):
        return None, EXIT_USAGE, [f"compile database not found: {db} (configure first)"]
    with open(db, encoding="utf-8") as fh:
        entries = json.load(fh)
    tus = {}
    for e in entries:
        f = e.get("file", "")
        if not f.lower().endswith(TU_EXTS):
            continue  # e.g. RC entries are no clang input
        norm = (os.path.normpath(f) if os.path.isabs(f)
                else os.path.normpath(os.path.join(e.get("directory", ""), f)))
        tus[norm] = True
        if not os.path.isfile(norm):
            return (None, EXIT_USAGE,
                    [f"stale compile database: {norm} vanished (reconfigure first)"])
    csd = find_clang_scan_deps(contract.clang_scan_deps)
    if csd != "clang-scan-deps" and not os.path.isfile(csd):
        return None, EXIT_USAGE, [f"clang-scan-deps not found: {csd}"]
    # CMake's C++ module scanning flags reference .modmap files that only
    # exist after a build; strip them so the scan works on a fresh tree.
    scrubbed = []
    for e in entries:
        e = dict(e)
        parts = [p for p in e.get("command", "").split()
                 if not p.startswith("@CMakeFiles")
                 and p not in MODULE_SCAN_FLAGS
                 and not p.startswith("-fmodule-mapper")
                 and not p.startswith("-fdeps-format")]
        e["command"] = " ".join(parts)
        scrubbed.append(e)
    tmp = os.path.join(os.environ.get("TEMP", os.path.dirname(db)),
                       "check_arch_scan_deps_db.json")
    with open(tmp, "w") as fh:
        json.dump(scrubbed, fh)
    r = subprocess.run([csd, "-compilation-database=" + tmp],
                       capture_output=True, text=True, check=False)
    if r.returncode != 0 and not r.stdout:
        return (None, EXIT_USAGE,
                [f"clang-scan-deps failed ({csd}): {r.stderr.strip()[-500:]}"])
    deps = parse_make_deps(r.stdout)
    scanned = {os.path.normpath(tu) for tu in deps}
    unscanned = sorted(t for t in tus if t not in scanned)
    if unscanned:
        rel = [os.path.relpath(t, repo) for t in unscanned]
        return (None, EXIT_VIOLATIONS,
                [f"incomplete scan: {len(unscanned)} of {len(tus)} translation units "
                 "not scanned"] + ["  " + p for p in rel])
    norm_to_file = {os.path.normpath(p): p for p in files}
    adj = {p: set() for p in files}
    file_mod = {}
    for p, rel in files.items():
        file_mod[p] = contract.module_of(rel)
    mod_edges = {}
    ext_counts = {}
    forbidden_hits = []
    for tu, incs in deps.items():
        tp = norm_to_file.get(os.path.normpath(tu))
        if tp is None:
            continue  # a TU outside the scan roots is not part of this contract
        a = file_mod[tp]
        for inc in sorted(incs):
            ip = norm_to_file.get(os.path.normpath(inc))
            if ip is None:
                rel = os.path.relpath(inc, repo).replace(os.sep, "/") \
                    if inc.startswith(repo) else inc.replace("\\", "/")
                first = rel.split("/")[0]
                if first in contract.external:
                    ext_counts[first] = ext_counts.get(first, 0) + 1
                elif any(rel.startswith(x) for x in contract.forbidden):
                    forbidden_hits.append(f"{os.path.relpath(tp, repo)}: must not include "
                                          f"{rel} (forbidden dependency)")
                continue
            if ip == tp:
                continue
            adj[tp].add(ip)
            b = file_mod[ip]
            if a is not None and b is not None and a != b:
                mod_edges[(a, b)] = mod_edges.get((a, b), 0) + 1
    return ((adj, file_mod, mod_edges, ext_counts, forbidden_hits, []), 0, [])


def check_nesting(contract):
    """THE NESTING LAW (GameBits): a module must not depend on its own
    child module - the deeper namespace uses its shallower ancestors,
    never the reverse (foo must not include foo/bar)."""
    out = []
    for a, b in sorted(contract.edges):
        if b.startswith(a + "/"):
            out.append(f"NESTING LAW: {a} must not depend on its child module {b} - "
                       "the deeper module uses its ancestors, never the reverse")
    return out


def check_edges(contract, files, file_mod, mod_edges):
    """DIRECTION (default deny) + CLASSIFICATION over the measured edges."""
    violations = []
    if contract.unclassified == "error":
        for p in sorted(files, key=lambda k: files[k]):
            if file_mod[p] is None:
                violations.append(
                    f"unclassified module: {files[p]} (matches no declared module)")
    for (a, b), n in sorted(mod_edges.items()):
        if a is None or b is None:
            continue  # unclassified sides are flagged per policy above
        if (a, b) not in contract.edges:
            violations.append(f"{a} must not include {b} (x{n}; edge not declared)")
    return violations


def check_stack(contract):
    """THE LAYER STACK: even a DECLARED edge may not run against the stack
    (a higher layer may not be consumed from below it)."""
    out = []
    for a, b in sorted(contract.edges):
        la, lb = contract.layer_of.get(a), contract.layer_of.get(b)
        if la and lb and contract.rank[lb] > contract.rank[la]:
            out.append(f"{a} (layer {la}) must not depend on {b} (layer {lb}) - "
                       "the layer stack flows one way")
    return out


def check_cycles(contract, files, adj):
    out = []
    for comp in scc(list(files), adj):
        if len(comp) > 1:
            names = sorted(files[x] for x in comp)
            out.append("include cycle: " + " <-> ".join(names))
    return out


def check_namespaces(contract, files, file_mod):
    ns_re = re.compile(r"^\s*namespace\s+(" + re.escape(contract.ns_root)
                       + r"(?:::[A-Za-z_]\w*)*)\s*\{", re.MULTILINE)
    out = []
    for p in sorted(files, key=lambda k: files[k]):
        mod = file_mod[p]
        if mod is None:
            continue
        if mod == contract.ns_module:
            expected = contract.ns_root
        elif mod.startswith(contract.ns_module + "/"):
            expected = contract.ns_root + "::" + mod[len(contract.ns_module) + 1:].replace(
                "/", "::")
        else:
            expected = contract.ns_root + "::" + mod.replace("/", "::")
        with open(p, encoding="utf-8", errors="replace") as fh:
            opens = set(ns_re.findall(fh.read()))
        opens_own = any(o == expected or o.startswith(expected + "::") for o in opens)
        ext = os.path.splitext(p)[1]
        if not opens_own and (ext in contract.ns_require_all or opens):
            out.append(f"{files[p]}: never opens namespace {expected}")
    return out


def check_diagram(contract, text):
    """The drift gate: the marked block section must equal the generated one."""
    a = text.find(DIAGRAM_BEGIN)
    b = text.find(DIAGRAM_END)
    if a < 0 or b < 0:
        return []  # no marked diagram = visualization not maintained by the tool
    shown = text[a + len(DIAGRAM_BEGIN):b].strip()
    want = contract.marked_diagram().strip()
    if shown != want:
        return ["DIAGRAM DRIFT: the diagram section is stale "
                "(run check_arch.py --diagram)"]
    return []


def write_diagram(doc_path, contract, text):
    a = text.find(DIAGRAM_BEGIN)
    b = text.find(DIAGRAM_END)
    if a < 0 or b < 0:
        return False
    text = (text[:a + len(DIAGRAM_BEGIN)] + "\n" + contract.marked_diagram()
            + "\n" + text[b:])
    with open(doc_path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)
    return True


def scc(nodes, adj):
    """Tarjan, iterative: strongly-connected components over the file graph."""
    idx, low, on, stack, out = {}, {}, set(), [], []
    counter = [0]
    for start in nodes:
        if start in idx:
            continue
        work = [(start, iter(sorted(adj.get(start, ()))))]
        idx[start] = low[start] = counter[0]
        counter[0] += 1
        stack.append(start)
        on.add(start)
        while work:
            v, it = work[-1]
            advanced = False
            for w in it:
                if w not in idx:
                    idx[w] = low[w] = counter[0]
                    counter[0] += 1
                    stack.append(w)
                    on.add(w)
                    work.append((w, iter(sorted(adj.get(w, ())))))
                    advanced = True
                    break
                elif w in on:
                    low[v] = min(low[v], idx[w])
            if advanced:
                continue
            work.pop()
            if work:
                pv = work[-1][0]
                low[pv] = min(low[pv], low[v])
            if low[v] == idx[v]:
                comp = []
                while True:
                    w = stack.pop()
                    on.discard(w)
                    comp.append(w)
                    if w == v:
                        break
                out.append(comp)
    return out


# ---------------------------------------------------------------- derive ---
def derive_scan(repo, roots, extensions, excludes):
    files = {}
    for root in roots:
        base = os.path.normpath(os.path.join(repo, root))
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(
                d for d in dirnames
                if not any(x in os.path.join(dirpath, d) for x in excludes))
            for name in sorted(filenames):
                if os.path.splitext(name)[1] in extensions:
                    p = os.path.normpath(os.path.join(dirpath, name))
                    files[p] = os.path.relpath(p, repo).replace(os.sep, "/")
    return files


def topo_leaf_first(units, mod_edges):
    """(order, cyclic leftover): dependencies first (Kahn, alpha tie-break)."""
    deps = {u: set() for u in units}
    for a, b in mod_edges:
        deps.setdefault(a, set()).add(b)
    order, remaining = [], set(units)
    while remaining:
        ready = sorted(u for u in remaining if not (deps.get(u, set()) & remaining))
        if not ready:
            return order, sorted(remaining)
        for u in ready:
            order.append(u)
            remaining.discard(u)
    return order, []


def derive(repo, roots, extensions, excludes):
    """ANALYZE: the modules as they are + the layers as they measure + the
    dependencies as they are -> a draft ARCHITECTURE.md."""
    files = derive_scan(repo, roots, extensions, excludes)
    warnings = []
    if not files:
        return [], [f"no sources under {roots} - nothing to derive"]
    key_of = {}
    rootless = 0
    for p, rel in files.items():
        root = next((r for r in roots if rel.startswith(r + "/")), None)
        if root is None:
            rootless += 1
            continue
        sub = os.path.dirname(rel[len(root) + 1:]).replace("\\", "/")
        key_of[p] = sub if sub != "" else root.split("/")[-1]
    if rootless:
        warnings.append(f"{rootless} file(s) sit outside the scan roots - not derived")
    units = sorted({k for k in key_of.values() if k})
    mod_edges = set()
    ext = set()
    unresolved = 0
    for p, rel in files.items():
        a = key_of.get(p)
        if a is None:
            continue
        with open(p, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        for m in INCLUDE_RE.finditer(text):
            kind, inc = m.group(1), m.group(2)
            t = None
            if kind == '"':
                cand = os.path.normpath(os.path.join(os.path.dirname(p), inc))
                t = cand if cand in files else None
            if t is None:
                for root in roots:
                    cand = os.path.normpath(os.path.join(repo, root, inc))
                    if cand in files:
                        t = cand
                        break
            if t is None:
                first = inc.split("/")[0]
                if "/" in inc and not inc.startswith((".", "..")) and first not in roots:
                    ext.add(first)
                elif kind == '"':
                    unresolved += 1
                continue
            b = key_of.get(t)
            if a is not None and b is not None and a != b:
                mod_edges.add((a, b))
    # the layer proposal: one layer per first path segment, stacked by the
    # measured segment-level dependency order (bottom-up)
    segs = sorted({u.split("/")[0] for u in units})
    seg_edges = sorted({(a.split("/")[0], b.split("/")[0])
                        for a, b in mod_edges if a.split("/")[0] != b.split("/")[0]})
    stack, cyclic_layers = topo_leaf_first(segs, seg_edges)
    full_stack = stack + cyclic_layers
    if cyclic_layers:
        warnings.append("layer cycle among " + ", ".join(cyclic_layers)
                        + " - fix the cycle first; the stack stops at the cycle")
    order, cyclic = topo_leaf_first(units, sorted(mod_edges))
    if cyclic:
        warnings.append("cycle among " + ", ".join(cyclic)
                        + " - fix the cycle first; the edges are still emitted, "
                          "the order stops at the cycle")

    arch = []
    knobs = [("scan", ", ".join(roots)), ("extensions", ", ".join(extensions)),
             ("exclude", ", ".join(DEFAULT_EXCLUDES)), ("unclassified", "error")]
    if ext:
        knobs.append(("external", ", ".join(sorted(ext))))
    knobs += [("forbidden", ""), ("quotes_relative", "true"), ("repo_fallback", "false"),
              ("cycles", "true"), ("namespace_root", ""), ("namespace_module", ""),
              ("namespace_require", "")]
    for k, v in knobs:
        arch.append(f"{k}: {v}".rstrip())
    arch.append("")
    for lay in full_stack:
        arch.append(f"layer {lay}")
    arch.append("")
    for u in order + cyclic:
        arch.append(f"module {u} in {u.split('/')[0]}")
    arch.append("")
    for a, b in sorted(mod_edges):
        arch.append(f"allow {a} -> {b}")

    out = ["# Architecture", "",
           "<!-- GENERATED by check_arch.py --derive - the MEASURED architecture.",
           "     REVIEW AND TIGHTEN: every `allow` edge below is what the code does",
           "     TODAY; delete the edges that should not exist and reshape the",
           "     layer stack, then keep this file as the checked-in contract. -->",
           "", "## The machine contract", "", "```arch"]
    out.extend(arch)
    out.append("```")
    out.append("")
    out.append("## The measured order (leaf-first)")
    out.append("")
    out.append("The dependency order as measured - dependencies first:")
    out.append("")
    out.append("```")
    out.extend(order)
    out.append("```")
    out.append("")
    out.append("## The diagram")
    out.append("")
    out.append(DIAGRAM_BEGIN)
    out.append(Contract("# Architecture\n\n```arch\n" + "\n".join(arch) + "\n```\n",
                        "draft").marked_diagram())
    out.append(DIAGRAM_END)
    out.append("")
    if unresolved:
        warnings.append(f"{unresolved} quoted include(s) did not resolve - "
                        "re-check after adoption")
    return out, warnings


# ------------------------------------------------------------------ core ---
def collect(repo, doc_path, report=False):
    """The CHECK step: the code against the contract.
    -> (exit code, human lines, machine summary)."""
    try:
        resolved, text = load_contract_text(repo, doc_path)
        contract = Contract(text, resolved)
    except ContractError as e:
        return EXIT_USAGE, [f"contract error: {e}"], {
            "code": EXIT_USAGE, "error": str(e), "violations": None}
    files = scan_files(contract, repo)
    if not files:
        msg = f"skip: no sources under {contract.roots} (contract {resolved})"
        return EXIT_SKIP, [msg], {"code": EXIT_SKIP, "doc": resolved, "files": 0,
                                  "violations": None, "error": msg}
    if contract.graph == "scan-deps":
        graph, gcode, glines = build_graph_scan_deps(contract, repo, files)
        if graph is None:
            return gcode, glines, {"code": gcode, "doc": resolved,
                                   "violations": glines, "error": glines[0]}
        adj, file_mod, mod_edges, ext_counts, forbidden_hits, unresolved = graph
    else:
        adj, file_mod, mod_edges, ext_counts, forbidden_hits, unresolved = build_graph(
            contract, repo, files)
    violations = list(forbidden_hits)
    violations += check_stack(contract)
    violations += check_edges(contract, files, file_mod, mod_edges)
    if contract.nesting == "error":
        violations += check_nesting(contract)
    if contract.cycles:
        violations += check_cycles(contract, files, adj)
    if contract.ns_root:
        violations += check_namespaces(contract, files, file_mod)
    if contract.unresolved_policy == "error" and unresolved:
        violations += [f"unresolved include: {u}" for u in unresolved]
    violations += check_diagram(contract, text)

    summary = {
        "code": None, "doc": resolved, "graph": contract.graph, "files": len(files),
        "modules": sorted(contract.modules), "layer_stack": list(contract.stack),
        "declared_edges": [list(e) for e in sorted(contract.edges)],
        "measured_edges": [[a, b, n] for (a, b), n in sorted(mod_edges.items())],
        "external": ext_counts, "violations": violations, "clean": not violations,
    }
    lines = []
    if report:
        if contract.stack:
            lines.append("layer stack (bottom-up): " + " <- ".join(contract.stack))
            for lay in contract.stack:
                members = sorted(m for m in contract.modules
                                 if contract.layer_of.get(m) == lay)
                if members:
                    lines.append(f"  {lay}: " + ", ".join(members))
            floating = sorted(m for m in contract.modules if m not in contract.layer_of)
            if floating:
                lines.append("  (unlayered): " + ", ".join(floating))
        lines.append("declared edges: "
                     + ", ".join(f"{a}->{b}" for a, b in sorted(contract.edges)))
        lines.append("measured module edges:")
        for (a, b), n in sorted(mod_edges.items()):
            ok = "" if (a, b) in contract.edges else "  NOT DECLARED"
            lines.append(f"  {a} -> {b}  x{n}{ok}")
        if ext_counts:
            lines.append("external (allowed, reported): "
                         + ", ".join(f"{k} x{v}" for k, v in sorted(ext_counts.items())))
        if unresolved and contract.unresolved_policy != "error":
            lines.append("unresolved quoted includes (info only):")
            for u in unresolved[:10]:
                lines.append("  " + u)
    if violations:
        summary["code"] = EXIT_VIOLATIONS
        return (EXIT_VIOLATIONS,
                lines + [f"{len(violations)} architecture violation(s):"]
                + ["  " + v for v in violations], summary)
    tails = []
    if contract.stack:
        tails.append("layer stack holds")
    if contract.ns_root:
        tails.append("namespaces mapped")
    tail = (", " + ", ".join(tails)) if tails else ""
    lines.append(f"architecture clean ({len(files)} files, "
                 f"{len(mod_edges)} distinct module edges, no cycles{tail})")
    summary["code"] = EXIT_OK
    return EXIT_OK, lines, summary


def run_check(repo, doc_path, report=False):
    """The CHECK step (human lines). -> (exit code, lines)."""
    code, lines, _summary = collect(repo, doc_path, report=report)
    return code, lines


def selftest():
    """Offline PASS proof: every rule on synthetic fixtures. No repo writes."""
    import tempfile

    def write(base, files):
        for name, body in files.items():
            p = os.path.join(base, name)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(body)

    def contract(body):
        return "# Architecture\n\n```arch\n" + body.strip() + "\n```\n"

    def diagram_of(body):
        """The document with a marked (generated) diagram drill-down."""
        c = Contract(contract(body), "doc")
        return ("# Architecture\n\n" + DIAGRAM_BEGIN + "\n" + c.marked_diagram()
                + "\n" + DIAGRAM_END + "\n\n```arch\n" + body.strip() + "\n```\n")

    def run(files, doc, report=False):
        with tempfile.TemporaryDirectory(prefix="check_arch_self_") as td:
            write(td, files)
            write(td, {"ARCHITECTURE.md": doc})
            return run_check(td, os.path.join(td, "ARCHITECTURE.md"), report=report)

    ok = True

    def expect(name, got, want):
        nonlocal ok
        good = got == want
        ok = ok and good
        print(f"selftest {name}: {'PASS' if good else f'FAIL (got {got}, want {want})'}")

    KNOBS = "scan: src\n"
    # 1 declared edges are legal; the chain passes
    b1 = KNOBS + """
module base
module mid
module top
allow mid -> base
allow top -> mid
"""
    files1 = {"src/base/a.h": "#pragma once\n",
              "src/mid/m.h": '#include "base/a.h"\n',
              "src/top/t.h": '#include "mid/m.h"\n'}
    e, out = run(files1, contract(b1))
    expect("declared chain clean", (e, [v for v in out if "must" in v]), (0, []))

    # 2 default deny: an undeclared edge fires
    b2 = KNOBS + "module base\nmodule top\n"
    files2 = {"src/base/a.h": "#pragma once\n",
              "src/top/t.h": '#include "base/a.h"\n'}
    e, out = run(files2, contract(b2))
    expect("undeclared edge fires (default deny)",
           (e, any("top must not include base" in v for v in out)), (1, True))

    # 3 longest-prefix module mapping + upward refusal
    b3 = KNOBS + "module proj\nmodule proj/sub\nallow proj/sub -> proj\n"
    files3 = {"src/proj/root.h": '#include "proj/sub/s.h"\n',
              "src/proj/sub/s.h": "#pragma once\n"}
    e, out = run(files3, contract(b3))
    expect("longest-prefix mapping + upward refusal",
           (e, any("proj must not include proj/sub" in v for v in out)), (1, True))

    # 4 the layer stack: downward declared edges hold, upward ones violate
    b4 = (KNOBS + "layer base\nlayer mid\nlayer top\n"
          "module base/x in base\nmodule mid/y in mid\nmodule top/z in top\n"
          "allow top/z -> mid/y\nallow mid/y -> base/x\n")
    files4 = {"src/base/x/x.h": "#pragma once\n",
              "src/mid/y/y.h": '#include "base/x/x.h"\n',
              "src/top/z/z.h": '#include "mid/y/y.h"\n'}
    e, out = run(files4, contract(b4))
    expect("stacked layers: downward edges hold",
           (e, [v for v in out if "layer" in v and "must" in v]), (0, []))
    b4bad = (KNOBS + "layer base\nlayer mid\nlayer top\n"
             "module base/x in base\nmodule mid/y in mid\nmodule top/z in top\n"
             "allow mid/y -> base/x\nallow base/x -> mid/y\n")
    e, out = run(files4, contract(b4bad))
    expect("stacked layers: an UPWARD declared edge violates the stack",
           (e, any("base/x (layer base) must not depend on mid/y (layer mid)" in v
                   for v in out)), (1, True))
    b4typo = KNOBS + "layer base\nmodule x in middle\n"
    e, out = run({}, contract(b4typo))
    expect("layer typo = contract error",
           (e, any("undeclared layer" in v for v in out)), (2, True))

    # 5 unclassified policy
    b5 = KNOBS + "module known\n"
    e, out = run({"src/known/k.h": "#pragma once\n", "src/stray/s.h": "#pragma once\n"},
                 contract(b5 + "unclassified: error\n"))
    expect("unclassified=error flags the stray file",
           (e, any("unclassified module: src/stray/s.h" in v for v in out)), (1, True))
    e, out = run({"src/known/k.h": "#pragma once\n", "src/stray/s.h": "#pragma once\n"},
                 contract(b5 + "unclassified: ignore\n"))
    expect("unclassified=ignore stays silent",
           (e, [v for v in out if "unclassified" in v]), (0, []))

    # 6 cycles: file-level SCC
    e, out = run({"src/known/a.h": '#include "known/b.h"\n',
                  "src/known/b.h": '#include "known/a.h"\n'}, contract(b5))
    expect("cycle detected", (e, any("include cycle:" in v and "a.h" in v for v in out)),
           (1, True))

    # 7 resolution: rooted via scan roots, quoted-relative, repo fallback
    b7 = ("scan: src, include\nrepo_fallback: true\n"
          "module proj\nmodule proj/sub\nallow proj -> proj/sub\n")
    files7 = {"src/proj/rf.h": '#include "proj/sub/s.h"\n',
              "include/proj/rf2.h": '#include "proj/sub/s.h"\n',
              "src/proj/sub/s.h": "#pragma once\n"}
    e, out = run(files7, contract(b7))
    expect("resolution: rooted via scan roots + repo fallback",
           (e, [v for v in out if "must" in v]), (0, []))

    # 8 external reported, forbidden violates
    b8 = ("scan: src\nexternal: other\nforbidden: renderer\nmodule lib\n")
    e, out = run({"src/lib/l.h": '#include <other/x.h>\n#include "renderer/r.h"\n'
                  "#include <system.h>\n"}, contract(b8))
    expect("forbidden fires, external allowed",
           (e, any('must not include "renderer/r.h"' in v for v in out)), (1, True))

    # 9 the namespace gate
    b9 = (KNOBS + "namespace_root: proj\nnamespace_module: proj\nnamespace_require: .hpp\n"
          "module proj\nmodule proj/sub\nallow proj -> proj/sub\n")
    ns_files = {"src/proj/Root.hpp": "namespace proj {\nint f();\n}\n",
                "src/proj/sub/S.hpp": "namespace proj::sub::Deep {\nint g();\n}\n",
                "src/proj/impl.cpp": "namespace proj::sub {\nint g() { return 0; }\n}\n",
                "src/proj/plain.cpp": "int helper() { return 1; }\n"}
    e, out = run(ns_files, contract(b9))
    expect("namespace gate: own namespaces pass",
           (e, [v for v in out if "never opens" in v]), (0, []))
    ns_bad = dict(ns_files)
    ns_bad["src/proj/sub/S.hpp"] = "int g();\n"
    e, out = run(ns_bad, contract(b9))
    expect("namespace gate: header without its namespace fires",
           (e, any("src/proj/sub/S.hpp: never opens namespace proj::sub" in v for v in out)),
           (1, True))
    ns_bad2 = dict(ns_files)
    ns_bad2["src/proj/sub/impl.cpp"] = "namespace proj::other {\nint g() { return 0; }\n}\n"
    e, out = run(ns_bad2, contract(b9))
    expect("namespace gate: cpp opening a foreign namespace fires",
           (e, any("src/proj/sub/impl.cpp: never opens namespace proj::sub" in v for v in out)),
           (1, True))

    # 10 the diagram drift gate + the block layout shape
    e, out = run(files4, diagram_of(b4))
    expect("fresh generated diagram passes the drift gate",
           (e, [v for v in out if "DRIFT" in v]), (0, []))
    stale = diagram_of(b4).replace("block-beta", "block-beta\n    %% stale")
    e, out = run(files4, stale)
    expect("stale diagram fires", (e, any("DIAGRAM DRIFT" in v for v in out)), (1, True))
    d = Contract(contract(b4), "doc").marked_diagram()
    expect("diagrams: the drill-down (stack + per layer), pull spans, slack",
           "### the layers" in d and 'top["top"]:2' in d and 'mid["mid"]:2' in d
           and 'base["base"]' in d and "space" in d
           and '### layer base - the modules' in d and 'base_x["base/x"]' in d
           and 'top_z["top/z"]:2' in d, True)

    # 10b the drill-down into nested containers
    b10 = (KNOBS + "layer base\nmodule proj in base\nmodule proj/sub in base\n"
           "allow proj/sub -> proj\n")
    d10 = Contract(contract(b10), "doc").marked_diagram()
    expect("drill-down: container module resolves into its own diagram",
           "### layer base - the modules" in d10
           and "### proj - the nested modules" in d10
           and 'proj_sub["proj/sub"]' in d10, True)

    # 11 contract errors are loud (exit 2)
    e, out = run(files1, contract(b1 + "allow mid -> typo\n"))
    expect("edge to an undeclared module = contract error",
           (e, any("contract error" in v for v in out)), (2, True))
    e, out = run(files1, "# Architecture\n\nno arch block here\n")
    expect("missing arch block = contract error",
           (e, any("no ```arch block" in v for v in out)), (2, True))

    # 12 derive: measured contract + layer proposal; a cycle is reported
    with tempfile.TemporaryDirectory(prefix="check_arch_deriv_") as td:
        write(td, {"src/leaf/l.h": "#pragma once\n",
                   "src/mid/m.h": '#include "leaf/l.h"\n',
                   "src/top/t.h": '#include "mid/m.h"\n'})
        lines, warns = derive(td, ["src"], [".h"], DEFAULT_EXCLUDES)
        text = "\n".join(lines)
        expect("derive: layers + modules + edges in the draft",
               "layer leaf" in text and "module mid in mid" in text
               and "allow mid -> leaf" in text and "allow top -> mid" in text, True)
        expect("derive: no warnings on a clean DAG", warns, [])
        write(td, {"src/leaf/l.h": '#include "top/t.h"\n'})
        lines, warns = derive(td, ["src"], [".h"], DEFAULT_EXCLUDES)
        expect("derive: cycle reported", any("cycle" in w for w in warns), True)

    # 13 exit-code contract
    e, _ = run({"src/known/k.h": "#pragma once\n"}, contract(b5))
    expect("exit 0 on a clean tree", e, 0)
    e, _ = run({"src/stray/s.h": "#pragma once\n"}, contract(b5))
    expect("exit 1 on findings", e, 1)
    e, _ = run({}, contract(b5))
    expect("exit 77 on nothing to scan", e, 77)

    print("check_arch selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 2


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        prog="check_arch.py",
        description="the generic architecture gate (analyze, define, check; rules in "
                    "the arch block incl. the layer stack, the diagram generated + "
                    "drift-gated)")
    ap.add_argument("--repo", help="the repository root to check/derive "
                                   "(default: the directory above this script)")
    ap.add_argument("--doc", help="contract document (default: docs/ARCHITECTURE.md, "
                                  "ARCHITECTURE.md, tools/ARCHITECTURE.md)")
    ap.add_argument("--report", action="store_true",
                    help="also print the stack and the measured module graph")
    ap.add_argument("--json", action="store_true",
                    help="machine-readable output: the check summary as JSON")
    ap.add_argument("--diagram", action="store_true",
                    help="regenerate the marked diagram section from the rules")
    ap.add_argument("--derive", action="store_true",
                    help="ANALYZE: draft an ARCHITECTURE.md from the measured graph")
    ap.add_argument("--scan", help="derive only: comma-separated scan roots "
                                   "(default: every top-level source dir)")
    ap.add_argument("--out", help="derive only: write the draft to FILE (default: stdout)")
    ap.add_argument("--selftest", action="store_true",
                    help="offline PASS proof on synthetic fixtures")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    repo = args.repo or os.path.dirname(here)

    if args.derive:
        if args.scan:
            roots = [r.strip().replace("\\", "/") for r in args.scan.split(",") if r.strip()]
        else:
            roots = []
            for name in sorted(os.listdir(repo)):
                p = os.path.join(repo, name)
                if (not os.path.isdir(p) or name.startswith(".")
                        or any(x in name for x in DEFAULT_EXCLUDES)):
                    continue
                for dp, dn, fn in os.walk(p):
                    dn[:] = [d for d in dn if not any(x in d for x in DEFAULT_EXCLUDES)]
                    if any(os.path.splitext(f)[1] in DEFAULT_EXTS for f in fn):
                        roots.append(name.replace("\\", "/"))
                        break
        if not roots:
            print("derive: no source roots found (pass --scan DIRS)")
            return EXIT_SKIP
        lines, warns = derive(repo, roots, DEFAULT_EXTS, DEFAULT_EXCLUDES)
        text = "\n".join(lines) + "\n"
        if args.out:
            with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
            print(f"derive: draft written -> {args.out}")
        else:
            sys.stdout.write(text)
        for w in warns:
            print("warning: " + w, file=sys.stderr)
        return EXIT_OK

    try:
        doc_path, text = load_contract_text(repo, args.doc)
        contract = Contract(text, doc_path)
    except ContractError as e:
        print(f"contract error: {e}")
        return EXIT_USAGE

    if args.diagram:
        if write_diagram(doc_path, contract, text):
            print("the diagram regenerated from the rules -> " + doc_path)
            return EXIT_OK
        print(f"{doc_path}: no {DIAGRAM_BEGIN} markers to fill")
        return EXIT_USAGE

    code, lines, summary = collect(repo, args.doc, report=args.report)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        for line in lines:
            print(line)
    return code


if __name__ == "__main__":
    sys.exit(main())
