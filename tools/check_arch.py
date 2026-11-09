#!/usr/bin/env python3
# mypy: check-untyped-defs
"""check_arch.py - the generic architecture gate: analyze, define, check.

THE CANONICAL ENGINE of the check-arch skill: a repository delegates to it
or vendors a copy whose CHECK_ARCH_VERSION matches - compare the versions
before trusting a vendored gate.

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
  joins a layer with `module core/loader in core`. The stack is LAW:
  a declared edge that runs from a higher layer DOWN-REFERENCING is the
  only legal direction - an edge from layer X to a layer ABOVE X is a
  violation even though it is declared. Modules without a layer are
  outside the stack (unconstrained by it).

THE THREE-STEP MODEL:
  1. ANALYZE  `--derive` scans the sources, builds the measured include
     graph and emits a draft ARCHITECTURE.md: the modules AS THEY ARE,
     the layers AS THEY MEASURE, the dependencies AS THEY ARE (cycles
     reported for fixing first). Vendored / non-source directories
     (third_party, vendor, external, deps, downloads, packages, ports,
     vcpkg) are QUARANTINED - never architecture, always noise.
  2. DEFINE   the reviewed (and tightened) document becomes the checked-in
     contract: "this is the architecture".
  3. CHECK    the default mode proves the code against that contract on
     every run: DIRECTION (only DECLARED edges), STACK (layer order),
     CLASSIFICATION (every file maps to a declared module), CYCLES
     (file-level SCC), DIAGRAM DRIFT, and - opt-in - NAMESPACES.

THE `arch` BLOCK (the whole contract; 100% generic - the engine knows no
repo, no layer name, no namespace):

  ```arch
  scan: src                      # comma-separated roots (repo-relative);
                                 # a root that does not exist is an error
  extensions: .hpp, .cpp         # scanned source extensions (leading '.')
  exclude: build, _deps, .git    # excluded directories: a NAME (fnmatch
                                 # globs, e.g. cmake-build-*) matches that
                                 # directory anywhere below a root; a PATH
                                 # (with '/') matches that repo-relative
                                 # directory - never a substring
  unclassified: error            # error | ignore - files off the module map
  external: libfoo, libbar       # other-repo includes (reported, allowed)
  forbidden:                     # hard violations (e.g. sibling repos)
  quotes_relative: true          # quoted includes resolve includer-relative
  repo_fallback: false           # ... then against the repo root
  quotes_resolve:                # explicit chain (overrides the 2x2 above):
                                 # includer | root | includer-then-root |
                                 # root-then-includer
  within_layer: deny             # deny | allow | warn - undeclared edges
                                 # between modules of the SAME layer
  smells: report                 # off | report | warn - the refactoring-
                                 # catalog smell proxies (report: in
                                 # --report/--json only; warn: also as
                                 # warning lines; never findings)
  smell_limits:                  # fan=8 mutual=3 files=12 lines=1000
  kind: application              # application | library - a library's
                                 # zero-consumer modules are public API:
                                 # fan-in smells (orphan, island) and
                                 # module-volume (large module) are
                                 # relaxed; volume is judged per FILE
                                 # (header_size) in both kinds
  header_size: warn              # report | warn | error - THE HEADER-SIZE
                                 # RULE (both kinds, independent of
                                 # smells): one file over the lines limit
                                 # is bloat - warn by default, error once
                                 # the repo's bloat queue is drained
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

THE EXACT GRAPH (`graph: scan-deps`): clang-scan-deps over the compile
database supplies the preprocessor truth per translation unit -
conditional includes included - and the include scan still supplies the
direct project relationships (header-to-header edges, header-only
cycles); the rules check both, so a compiler-observed transitive
dependency is an edge like a direct one. The database is CONTRACTED: an
unreadable database, or one whose translation units vanished on disk, is
REFUSED (exit 2); every compiled TU must be scanned or the run fails
(exit 1); a blind graph (no TU inside the scanned roots) is refused
(exit 2) - a partial or degraded scan never passes silently. forbidden /
external match the include text in the direct graph and the repo-relative
path of compiler-observed files; the quote chain and `unresolved` apply
to the direct graph.

THE SMELL PROXIES (refactoring.guru catalog, the deterministic tier):
smells are REPORT-ONLY queue items for the review cadence - never findings
(the thresholds are judgment, a smell can never fail a run): island /
orphan module (Speculative Generality / Dead Code / Lazy Class),
pass-through module (Middle Man), mutual pair (Inappropriate Intimacy),
hub module (Shotgun Surgery / Divergent Change), large module (Large
Class, applications only). THE HEADER-SIZE RULE is different: a standing
rule in BOTH kinds, never switchable off - one file over the lines limit
is bloat, warn by default, promotable to error once the repo's bloat
queue is drained. The AST-level catalog smells (Long Method, Long
Parameter List, Data Clumps, Primitive Obsession, Feature Envy, Switch
Statements, ...) belong to the lint matrix (clang-tidy / cppcheck), not
this tool; the rest of the catalog is review judgment.

Exit codes: 0 clean - 1 violations - 2 config/usage error - 77 skipped.
Usage: python check_arch.py [--repo DIR] [--doc FILE] [--report] [--json]
                            [--diagram] [--derive [--scan DIRS] [--out FILE]]
                            [--selftest] [--version] [-h]
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EXIT_OK = 0
EXIT_VIOLATIONS = 1
EXIT_USAGE = 2
EXIT_SKIP = 77

CHECK_ARCH_VERSION = "1.5.0"

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]', re.MULTILINE)
ARCH_BLOCK_RE = re.compile(r"```arch[ \t]*\r?\n(.*?)```", re.DOTALL)
DIAGRAM_BEGIN = "<!-- check_arch: diagram BEGIN -->"
DIAGRAM_END = "<!-- check_arch: diagram END -->"
DEFAULT_EXTS = [".hpp", ".cpp", ".h", ".c"]
DEFAULT_EXCLUDES = ["build", "_deps", ".git"]
DOC_CANDIDATES = ["docs/ARCHITECTURE.md", "ARCHITECTURE.md", "tools/ARCHITECTURE.md"]
VSWHERE = os.path.join(
    os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
    "Microsoft Visual Studio",
    "Installer",
    "vswhere.exe",
)
# compile-database entries the scanner reads: C/C++ translation units only
TU_EXTS = (".cpp", ".c", ".cc", ".cxx")
# CMake's C++ module scanning flags reference .modmap files that only exist
# after a build; the scan strips them so it works on a fresh tree
MODULE_SCAN_FLAGS = ("-fmodules-ts", "-MD")
NAMESPACE_RE = re.compile(r"^[A-Za-z_]\w*(?:::[A-Za-z_]\w*)*$")
# the quote chains: where a quoted include is looked up, in order
QUOTE_STEPS: dict[str, tuple[str, ...]] = {
    "": (),
    "includer": ("includer",),
    "root": ("root",),
    "includer-then-root": ("includer", "root"),
    "root-then-includer": ("root", "includer"),
}

KNOB_KEYS = {
    "scan",
    "extensions",
    "exclude",
    "unclassified",
    "external",
    "forbidden",
    "quotes_relative",
    "repo_fallback",
    "quotes_resolve",
    "within_layer",
    "cycles",
    "graph",
    "db",
    "clang_scan_deps",
    "nesting",
    "unresolved",
    "smells",
    "smell_limits",
    "kind",
    "header_size",
    "namespace_root",
    "namespace_module",
    "namespace_require",
}

# derive quarantines these directory names - vendored / non-source trees are
# never architecture (ecosystem convention, not any repo's naming)
VENDOR_DIRS = {
    "third_party",
    "thirdparty",
    "vendor",
    "external",
    "deps",
    "downloads",
    "packages",
    "ports",
    "vcpkg",
    "_deps",
}

# the smell-proxy thresholds and their documented defaults (smell_limits)
SMELL_LIMIT_DEFAULTS = {"fan": 8, "mutual": 3, "files": 12, "lines": 1000}


class ContractError(Exception):
    """A contract problem the user must fix (exit 2)."""


def _norm_root(root):
    """A scan root as written -> its normalized repo-relative posix form
    (`./src/` and `src\\` both become `src`): module stripping compares
    against it, so every spelling must classify alike."""
    return posixpath.normpath(root.replace("\\", "/"))


def _check_extensions(path, key, values):
    """Extensions start with '.': `hpp` would match no file at all and turn
    the gate into a silent skip."""
    bad = [x for x in values if len(x) < 2 or not x.startswith(".")]
    if bad:
        raise ContractError(f"{path}: {key}: each entry starts with '.', got {bad}")


def _safe_id(path):
    return re.sub(r"[^A-Za-z0-9_]", "_", path)


class Contract:
    """The parsed architecture contract (the `arch` block of the document)."""

    def __init__(self, text, path):
        self.path = path
        blocks = ARCH_BLOCK_RE.findall(text)
        if not blocks:
            raise ContractError(f"{path}: no ```arch block (the machine contract)")
        knobs: dict[str, str] = {}
        layers: list[str] = []
        modules: list[tuple[str, str | None]] = []  # (path, layer-or-None)
        edges: list[tuple[str, str]] = []
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
                    key = k.strip()
                    if key not in KNOB_KEYS:
                        raise ContractError(
                            f"{path}: unknown knob {key!r} (known: {sorted(KNOB_KEYS)})"
                        )
                    if key in knobs:
                        # last-wins would silently drop the first declaration
                        raise ContractError(
                            f"{path}: duplicate knob {key!r} (declare each knob once)"
                        )
                    knobs[key] = v.strip()
                    continue
                raise ContractError(
                    f"{path}: cannot parse contract line: {raw.strip()!r}"
                )

        dupes = sorted({layer for layer in layers if layers.count(layer) > 1})
        if dupes:
            raise ContractError(f"{path}: duplicate layer declarations {dupes}")
        self.stack = layers  # bottom-up: first = the bottom
        self.rank = {name: i for i, name in enumerate(layers)}
        dupes = sorted({p for p, _ in modules if [x for x, _ in modules].count(p) > 1})
        if dupes:
            raise ContractError(f"{path}: duplicate module declarations {dupes}")
        self.modules = {p for p, _ in modules}
        self.layer_of: dict[str, str] = {}
        for p, lay in modules:
            if lay is not None and lay not in self.rank:
                raise ContractError(
                    f"{path}: module {p} in undeclared layer {lay!r} "
                    f"(declare it with 'layer')"
                )
            if lay is not None:
                self.layer_of[p] = lay
        self.edges = set(edges)
        for a, b in edges:
            missing = [x for x in (a, b) if x not in self.modules]
            if missing:
                raise ContractError(
                    f"{path}: edge {a} -> {b} references undeclared "
                    f"module(s) {missing} (declare them with 'module')"
                )
        selfloops = sorted({a for a, b in edges if a == b})
        if selfloops:
            raise ContractError(
                f"{path}: self-edges are implicit, drop them: {selfloops}"
            )

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

        self.roots = [_norm_root(r) for r in lst("scan", [])]
        if not self.roots:
            raise ContractError(
                f"{path}: the contract needs 'scan:' (at least one root)"
            )
        self.extensions = lst("extensions", DEFAULT_EXTS)
        _check_extensions(path, "extensions", self.extensions)
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
            raise ContractError(
                f"{path}: namespace_root and namespace_module go together"
            )
        if self.ns_root and not NAMESPACE_RE.match(self.ns_root):
            raise ContractError(
                f"{path}: namespace_root: a C++ namespace expected "
                f"(e.g. proj or proj::core), got {self.ns_root!r}"
            )
        if self.ns_module and not any(
            m == self.ns_module or m.startswith(self.ns_module + "/")
            for m in self.modules
        ):
            raise ContractError(
                f"{path}: namespace_module {self.ns_module!r} is neither a "
                "declared module nor the path prefix of one"
            )
        self.ns_require_all = lst("namespace_require", [])
        _check_extensions(path, "namespace_require", self.ns_require_all)
        if self.ns_require_all and not self.ns_root:
            raise ContractError(
                f"{path}: namespace_require needs namespace_root and "
                "namespace_module (the gate it tightens)"
            )
        # rooted-include bases per repository (see _prefix_bases)
        self.prefix_cache: dict[str, dict[str, list[str]]] = {}

        self.graph = knobs.get("graph") or "regex"
        if self.graph not in ("regex", "scan-deps"):
            raise ContractError(
                f"{path}: graph: regex|scan-deps expected, got {self.graph!r}"
            )
        self.db = knobs.get("db") or ""
        if self.graph == "scan-deps" and not self.db:
            raise ContractError(
                f"{path}: graph: scan-deps needs 'db:' (the compile database)"
            )
        self.clang_scan_deps = (
            knobs.get("clang_scan_deps")
            or os.environ.get("CHECK_ARCH_CLANG_SCAN_DEPS")
            or ""
        )
        self.nesting = knobs.get("nesting") or "off"
        if self.nesting not in ("off", "error"):
            raise ContractError(
                f"{path}: nesting: off|error expected, got {self.nesting!r}"
            )
        self.unresolved_policy = knobs.get("unresolved") or "info"
        if self.unresolved_policy not in ("info", "error"):
            raise ContractError(f"{path}: unresolved: info|error expected")
        self.quotes_resolve = knobs.get("quotes_resolve") or None
        if self.quotes_resolve and self.quotes_resolve not in (
            "includer",
            "root",
            "includer-then-root",
            "root-then-includer",
        ):
            raise ContractError(
                f"{path}: quotes_resolve: includer | root | "
                "includer-then-root | root-then-includer expected"
            )
        self.within_layer = knobs.get("within_layer") or "deny"
        if self.within_layer not in ("deny", "allow", "warn"):
            raise ContractError(f"{path}: within_layer: deny|allow|warn expected")
        self.smells = knobs.get("smells") or "report"
        if self.smells not in ("off", "report", "warn"):
            raise ContractError(f"{path}: smells: off|report|warn expected")
        self.smell_limits = dict(SMELL_LIMIT_DEFAULTS)
        lim = knobs.get("smell_limits")
        if lim:
            for item in lim.replace(",", " ").split():
                if "=" not in item:
                    raise ContractError(
                        f"{path}: smell_limits: key=value expected, got {item!r}"
                    )
                k, v = item.split("=", 1)
                if k not in SMELL_LIMIT_DEFAULTS:
                    raise ContractError(
                        f"{path}: smell_limits: unknown limit {k!r} "
                        f"(known: {sorted(SMELL_LIMIT_DEFAULTS)})"
                    )
                if not v.isdigit():
                    raise ContractError(
                        f"{path}: smell_limits: {k}: integer expected, got {v!r}"
                    )
                self.smell_limits[k] = int(v)
        self.kind = knobs.get("kind") or "application"
        if self.kind not in ("application", "library"):
            raise ContractError(
                f"{path}: kind: application|library expected, got {self.kind!r}"
            )
        self.header_size = knobs.get("header_size") or "warn"
        if self.header_size not in ("report", "warn", "error"):
            raise ContractError(
                f"{path}: header_size: report|warn|error expected, "
                f"got {self.header_size!r}"
            )

    def quote_chain(self):
        """How a quoted include resolves, in order: 'includer' = relative to
        the includer's directory, 'root' = against the repo root. The
        explicit `quotes_resolve` knob wins; else the 2x2 of
        quotes_relative/repo_fallback."""
        if self.quotes_resolve:
            return self.quotes_resolve
        if self.relative_quotes and self.repo_fallback:
            return "includer-then-root"
        if self.relative_quotes:
            return "includer"
        if self.repo_fallback:
            return "root"
        return ""

    def module_of(self, rel):
        """File (posix, scan-root-stripped) -> declared module (longest prefix)."""
        best = None
        for m in self.modules:
            if (rel == m or rel.startswith(m + "/")) and (
                best is None or len(m) > len(best)
            ):
                best = m
        return best

    def _direct_children(self, m):
        return sorted(
            c
            for c in self.modules
            if c.startswith(m + "/") and "/" not in c[len(m) + 1 :]
        )

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
        deps = {
            m: {b for (a, b) in self.edges if a == m and b in group and b != m}
            for m in group
        }
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
        by_level: dict[int, list[str]] = {}
        for m in sorted(members):
            by_level.setdefault(level[m], []).append(m)
        return [
            (None, [(m, self._pull_weight(m)) for m in by_level[lvl]])
            for lvl in sorted(by_level, reverse=True)
        ]

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
        sections: list[tuple[str, str]] = []
        # --- the top: the layer stack (each block = one layer)
        rows: list[tuple[str | None, list[tuple[str, int]]]] = []
        for lay in reversed(self.stack):
            if any(self.layer_of.get(m) == lay for m in self.modules):
                rows.append((None, [(lay, self._layer_pull(lay))]))
        unlayered = sorted(
            m for m in self.modules if m not in self.layer_of and self._root_module(m)
        )
        if unlayered:
            rows.append(("unlayered", [(m, self._pull_weight(m)) for m in unlayered]))
        if rows:
            sections.append(("the layers", self._block_text(rows)))
        # --- each layer resolves into its (top-level) modules
        for lay in reversed(self.stack):
            members = sorted(
                m
                for m in self.modules
                if self.layer_of.get(m) == lay and self._root_module(m)
            )
            if members:
                sections.append(
                    (
                        f"layer {lay} - the modules",
                        self._block_text(self._fall_rows(members)),
                    )
                )
        if unlayered:
            sections.append(
                (
                    "unlayered - the modules",
                    self._block_text(self._fall_rows(unlayered)),
                )
            )
        # --- then each container module resolves into its children, and so on
        for m in self._container_modules():
            sections.append(
                (
                    f"{m} - the nested modules",
                    self._block_text(self._fall_rows(self._direct_children(m))),
                )
            )
        return sections

    def marked_diagram(self):
        """The whole marked section: every generated diagram of the drill-down."""
        return "\n\n".join(
            f"### {heading}\n\n```mermaid\n{text}\n```"
            for heading, text in self.diagram_sections()
        )


def load_contract_text(repo, doc_arg):
    if doc_arg:
        p = doc_arg if os.path.isabs(doc_arg) else os.path.join(repo, doc_arg)
    else:
        p = next(
            (
                os.path.join(repo, c)
                for c in DOC_CANDIDATES
                if os.path.isfile(os.path.join(repo, c))
            ),
            None,
        )
        if p is None:
            raise ContractError(
                "no contract document found (looked for "
                + ", ".join(DOC_CANDIDATES)
                + "; run check_arch.py --derive to draft one)"
            )
    if not os.path.isfile(p):
        raise ContractError(f"contract document not found: {p}")
    # newline="": the document's own line endings survive the round trip -
    # a CRLF working tree must not be silently normalized to LF
    with open(p, encoding="utf-8", newline="") as fh:
        return p, fh.read()


def _diagram_span(text):
    """(begin, end) offsets of the marked diagram section, or None when the
    document carries no markers. A half-present or reversed pair is a
    contract error: the drift gate would skip it in silence, and a rewrite
    would duplicate the text between the markers."""
    a = text.find(DIAGRAM_BEGIN)
    b = text.find(DIAGRAM_END)
    if a < 0 and b < 0:
        return None
    if a < 0 or b < a:
        raise ContractError(
            f"diagram markers malformed: {DIAGRAM_BEGIN} must precede {DIAGRAM_END}"
        )
    return a, b


def _excluded(rel_dir, patterns):
    """True when a directory is excluded. A pattern without '/' matches the
    directory NAME wherever it sits (fnmatch globs, case-sensitive); a
    pattern with '/' matches the repo-relative directory PATH. Never a
    substring: neither the checkout location nor an unrelated name such as
    `rebuild_queue` may hide sources from the gate."""
    name = rel_dir.rsplit("/", 1)[-1]
    for pattern in patterns:
        if "/" in pattern:
            if fnmatch.fnmatchcase(rel_dir, pattern.strip("/")):
                return True
        elif fnmatch.fnmatchcase(name, pattern):
            return True
    return False


def _walk_sources(repo, roots, extensions, excludes):
    """-> ({absolute path: repo-relative posix path}, [pruned repo-relative
    directories]): every source below the roots, excluded directories
    pruned whole."""
    files: dict[str, str] = {}
    pruned: list[str] = []
    for root in roots:
        base = os.path.normpath(os.path.join(repo, root))
        for dirpath, dirnames, filenames in os.walk(base):
            rel_dir = os.path.relpath(dirpath, repo).replace(os.sep, "/")
            keep = []
            for d in sorted(dirnames):
                rel = d if rel_dir == "." else f"{rel_dir}/{d}"
                if _excluded(rel, excludes):
                    pruned.append(rel)
                else:
                    keep.append(d)
            dirnames[:] = keep
            for name in sorted(filenames):
                if os.path.splitext(name)[1] in extensions:
                    p = os.path.normpath(os.path.join(dirpath, name))
                    files[p] = os.path.relpath(p, repo).replace(os.sep, "/")
    return files, pruned


def scan_files(contract, repo):
    """The contract's sources: {absolute path: repo-relative posix path}. A
    declared root that does not exist is a contract error - a typo would
    otherwise drop its sources from the gate in silence."""
    missing = [r for r in contract.roots if not os.path.isdir(os.path.join(repo, r))]
    if missing:
        raise ContractError(
            f"{contract.path}: scan root(s) {missing} do not exist under {repo}"
        )
    files, _ = _walk_sources(
        repo, contract.roots, contract.extensions, contract.exclude_dirs
    )
    return files


def _prefix_bases(contract, repo):
    """The rooted-include bases: {first path segment: [the scan roots holding
    a directory of that name]} - `proj/types.hpp` with src/proj/ resolves
    against src/. Roots keep their declaration order, so two roots sharing
    a directory name resolve identically on every run."""
    bases = contract.prefix_cache.get(repo)
    if bases is None:
        bases = {}
        for root in contract.roots:
            base = os.path.join(repo, root)
            if not os.path.isdir(base):
                continue
            for name in sorted(os.listdir(base)):
                if os.path.isdir(os.path.join(base, name)):
                    bases.setdefault(name, []).append(root)
        contract.prefix_cache[repo] = bases
    return bases


def resolve(contract, repo, inc, kind, from_file, files):
    """An include -> the scanned file it names, or None. The rooted bases
    come first (every root holding the first segment, in declaration
    order); a quoted include then walks the contract's quote chain - also
    when a rooted candidate does not exist."""
    if "/" in inc:
        for base in _prefix_bases(contract, repo).get(inc.split("/", 1)[0], ()):
            t = os.path.normpath(os.path.join(repo, base, inc))
            if t in files:
                return t
    if kind != '"':
        return None
    for step in QUOTE_STEPS[contract.quote_chain()]:
        if step == "includer":
            t = os.path.normpath(os.path.join(os.path.dirname(from_file), inc))
        else:
            t = os.path.normpath(os.path.join(repo, inc))
        if t in files:
            return t
    return None


def classify(contract, files):
    """{file: declared module or None}: the longest declared module prefix
    of the path without its scan-root segment, else with it (roots may be
    technical containers like src/, or architectural units like app/). One
    classifier serves both graph modes."""
    file_mod: dict[str, str | None] = {}
    for p, rel in files.items():
        stripped = rel
        for root in contract.roots:
            if root != "." and rel.startswith(root + "/"):
                stripped = rel[len(root) + 1 :]
                break
        file_mod[p] = contract.module_of(stripped) or contract.module_of(rel)
    return file_mod


def build_graph(contract, repo, files, observed=None):
    """The direct include graph -> (adj, file_mod, mod_edges, ext_counts,
    forbidden_hits, unresolved). `observed` (scan-deps mode) maps a file to
    the project files the compiler saw with it: the fallback for an include
    the resolver cannot bind - one candidate binds, several refuse."""
    adj: dict[str, set[str]] = {p: set() for p in files}
    file_mod = classify(contract, files)
    mod_edges: dict[tuple[str, str], int] = {}
    ext_counts: dict[str, int] = {}
    forbidden_hits: list[str] = []
    unresolved: list[str] = []
    for p, rel in files.items():
        with open(p, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        for m in INCLUDE_RE.finditer(text):
            kind, inc = m.group(1), m.group(2)
            if any(inc.startswith(x) for x in contract.forbidden):
                forbidden_hits.append(
                    f'{rel}: must not include "{inc}" (forbidden dependency)'
                )
                continue
            if any(
                inc.split("/")[0] == x or inc.startswith(x + "/")
                for x in contract.external
            ):
                key = inc.split("/")[0]
                ext_counts[key] = ext_counts.get(key, 0) + 1
                continue
            t = resolve(contract, repo, inc, kind, p, files)
            if t is None and observed is not None:
                suffix = "/" + inc.replace("\\", "/")
                candidates = sorted(
                    target
                    for target in observed.get(p, ())
                    if target.replace("\\", "/").endswith(suffix)
                )
                if len(candidates) > 1:
                    raise ContractError(
                        f'{rel}: ambiguous project include "{inc}" '
                        f"(observed candidates: {[files[x] for x in candidates]})"
                    )
                if candidates:
                    t = candidates[0]
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


def _make_rules(text):
    """Make output -> logical rules: backslash-continued lines joined."""
    text = text.replace("\\\r\n", " ").replace("\\\n", " ")
    return text[:-1].splitlines() if text.endswith("\\") else text.splitlines()


MAKE_CUT_RE = re.compile(r":(?:[ \t]|$)")
# a backslash run before an escaped blank (the blank already marked as NUL)
MAKE_BLANK_RE = re.compile(r"(\\*)\x00")


def _make_unescape(word):
    """Undo the emitter's escapes in a word whose escaped blanks the split
    marked as NUL: 2k+1 backslashes before a blank are k backslashes and a
    literal blank, a backslash before '#' escapes it, '$$' is '$'. Any
    other backslash is literal (Windows separators)."""
    if "\x00" in word:
        if "\\\x00" in word:  # a longer backslash run before a blank
            word = MAKE_BLANK_RE.sub(
                lambda m: "\\" * ((len(m.group(1)) + 1) // 2) + " ", word
            )
        else:
            word = word.replace("\x00", " ")
    if "\\#" in word:
        word = word.replace("\\#", "#")
    if "$$" in word:
        word = word.replace("$$", "$")
    return word


def parse_make_deps(text):
    """clang-scan-deps make output -> {tu: set(dependency paths)}, '/'-separated.

    One rule per translation unit, `target: tu dep dep ...`: it may span
    backslash-continued lines and carry several paths per line (short paths
    share a line), the target ends at the first ':' followed by a blank (a
    drive letter's ':' is not), the first prerequisite is the TU, and
    escaped blanks survive (`C:\\Program\\ Files\\...`). Rules naming the same
    TU merge."""
    deps: dict[str, set[str]] = {}
    for rule in _make_rules(text):
        cut = MAKE_CUT_RE.search(rule)
        if cut is None:
            continue
        paths = [
            (
                _make_unescape(w) if ("\x00" in w or "\\#" in w or "$$" in w) else w
            ).replace("\\", "/")
            for w in rule[cut.end() :].replace("\\ ", "\x00").split()
        ]
        if paths:
            tu = paths[0]
            deps.setdefault(tu, set()).update(p for p in paths[1:] if p != tu)
    return deps


def find_clang_scan_deps(explicit):
    """The scan tool: explicit knob / env override, then PATH, then the
    newest Visual Studio's LLVM component (vswhere) - a VS upgrade or
    edition change needs no repo edit."""
    if explicit:
        return explicit
    p = shutil.which("clang-scan-deps")
    if p:
        return p
    try:
        root = subprocess.check_output(
            [VSWHERE, "-latest", "-property", "installationPath"], text=True
        ).strip()
        cand = os.path.join(root, r"VC\Tools\Llvm\x64\bin\clang-scan-deps.exe")
        if os.path.isfile(cand):
            return cand
    except (OSError, subprocess.CalledProcessError):
        pass
    return "clang-scan-deps"


def _path_key(path):
    """A comparison key for a filesystem path: normalized, and case-folded
    where the filesystem is case-insensitive (Windows: `c:\\x` is `C:/X`)."""
    return os.path.normcase(os.path.normpath(path))


def _repo_relative(path, repo):
    """The repo-relative posix form of `path`, or None outside the repo."""
    try:
        rel = os.path.relpath(os.path.normpath(path), repo).replace(os.sep, "/")
    except ValueError:  # another drive (Windows)
        return None
    if rel == ".." or rel.startswith("../") or os.path.isabs(rel):
        return None
    return rel


def _keep_scan_arg(arg):
    """False for CMake's C++ module scanning arguments (see MODULE_SCAN_FLAGS)."""
    return not (
        arg.startswith(("@CMakeFiles", "-fmodule-mapper", "-fdeps-format"))
        or arg in MODULE_SCAN_FLAGS
    )


def _scrub_entry(entry):
    """A compile-database entry without module scanning arguments, in its own
    form: `arguments` stays a list, `command` stays a string."""
    e = dict(entry)
    if isinstance(e.get("arguments"), list):
        e["arguments"] = [
            a for a in e["arguments"] if not isinstance(a, str) or _keep_scan_arg(a)
        ]
    elif isinstance(e.get("command"), str):
        e["command"] = " ".join(a for a in e["command"].split() if _keep_scan_arg(a))
    return e


class _Observed:
    """{file: the project files the compiler saw in a TU with it}, built on
    first use per file - the include fallback rarely needs it, and building
    it eagerly costs O(headers^2) per translation unit."""

    def __init__(self, tu_sets, containing):
        self._tu_sets = tu_sets
        self._containing = containing
        self._cache: dict[str, set[str]] = {}

    def get(self, path, default=()):
        if path not in self._containing:
            return default
        seen = self._cache.get(path)
        if seen is None:
            seen = set()
            for index in self._containing[path]:
                seen |= self._tu_sets[index]
            seen.discard(path)
            self._cache[path] = seen
        return seen


def build_graph_scan_deps(contract, repo, files):
    """Complete compiler dependencies plus direct project include topology.

    -> (graph-6-tuple, error_code, error_lines): on success the tuple and
    (0, []); a refused database (missing, unreadable, vanished TUs, tool
    failure) is (None, EXIT_USAGE, [...]); a missing tool is
    (None, EXIT_SKIP, [...]); an incomplete scan (a compiled TU the scan
    missed) never passes silently: (None, EXIT_VIOLATIONS, [...])."""
    db = os.path.normpath(os.path.join(repo, contract.db))
    if not os.path.isfile(db):
        return None, EXIT_USAGE, [f"compile database not found: {db} (configure first)"]
    try:
        with open(db, encoding="utf-8") as fh:
            entries = json.load(fh)
    except (OSError, ValueError) as error:  # ValueError: bad JSON or bad UTF-8
        return None, EXIT_USAGE, [f"compile database unreadable: {db} ({error})"]
    if not isinstance(entries, list) or not all(
        isinstance(e, dict)
        and isinstance(e.get("file"), str)
        and isinstance(e.get("directory", ""), str)
        for e in entries
    ):
        return (
            None,
            EXIT_USAGE,
            [f"compile database malformed: {db} (a list of entries with 'file')"],
        )
    tus: dict[str, str] = {}  # path key -> the TU as the database names it
    scrubbed = []
    for e in entries:
        f = e["file"]
        if not f.lower().endswith(TU_EXTS):
            continue  # resource/assembler commands are no C/C++ scanner input
        norm = os.path.normpath(
            f if os.path.isabs(f) else os.path.join(e.get("directory", ""), f)
        )
        if not os.path.isfile(norm):
            return (
                None,
                EXIT_USAGE,
                [f"stale compile database: {norm} vanished (reconfigure first)"],
            )
        tus[_path_key(norm)] = norm
        scrubbed.append(_scrub_entry(e))
    csd = find_clang_scan_deps(contract.clang_scan_deps)
    try:
        with tempfile.TemporaryDirectory(prefix="check_arch_scan_deps_") as td:
            tmp = os.path.join(td, "compile_commands.json")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(scrubbed, fh)
            try:
                r = subprocess.run(
                    [csd, "-compilation-database=" + tmp],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
            except OSError:
                # a missing tool is a SKIP, never a traceback
                return (
                    None,
                    EXIT_SKIP,
                    [
                        (
                            f"skip: clang-scan-deps not found ({csd}) - install the VS "
                            "LLVM component, put it on PATH, or set clang_scan_deps:"
                        )
                    ],
                )
    except OSError as error:
        return None, EXIT_USAGE, [f"cannot stage the scanner database: {error}"]
    if r.returncode != 0:
        return (
            None,
            EXIT_USAGE,
            [f"clang-scan-deps failed ({csd}): {r.stderr.strip()[-500:]}"],
        )
    deps = parse_make_deps(r.stdout)
    scanned = {_path_key(tu) for tu in deps}
    unscanned = sorted(norm for key, norm in tus.items() if key not in scanned)
    if unscanned:
        return (
            None,
            EXIT_VIOLATIONS,
            [
                (
                    f"incomplete scan: {len(unscanned)} of {len(tus)} translation "
                    "units not scanned"
                )
            ]
            + ["  " + (_repo_relative(t, repo) or t) for t in unscanned],
        )
    by_key = {_path_key(p): p for p in files}
    file_mod = classify(contract, files)
    tu_sets: list[frozenset[str]] = []
    containing: dict[str, list[int]] = {}
    resolved_tus = 0
    mod_edges: dict[tuple[str, str], int] = {}
    ext_counts: dict[str, int] = {}
    forbidden_hits: list[str] = []
    for tu, incs in deps.items():
        project: set[str] = set()
        outside: list[str] = []
        for inc in incs:
            ip = by_key.get(_path_key(inc))
            if ip is None:
                outside.append(inc)
            else:
                project.add(ip)
        for ip in project:
            containing.setdefault(ip, []).append(len(tu_sets))
        tu_sets.append(frozenset(project))
        tp = by_key.get(_path_key(tu))
        if tp is None:
            continue  # a TU outside the scan roots is not part of this contract
        containing.setdefault(tp, []).append(len(tu_sets) - 1)
        resolved_tus += 1
        a = file_mod[tp]
        for inc in sorted(outside):
            rel = _repo_relative(inc, repo) or inc
            first = rel.split("/")[0]
            if first in contract.external:
                ext_counts[first] = ext_counts.get(first, 0) + 1
            elif any(rel.startswith(x) for x in contract.forbidden):
                forbidden_hits.append(
                    f"{files[tp]}: must not include {rel} (forbidden dependency)"
                )
        for ip in sorted(project):
            if ip == tp:
                continue
            b = file_mod[ip]
            if a is not None and b is not None and a != b:
                mod_edges[(a, b)] = mod_edges.get((a, b), 0) + 1
    # a degraded graph is never green: the database knows TUs, none of them
    # resolved into the scanned roots (e.g. a relative repo root or a
    # path-form mismatch between the database and the scan)
    if tus and resolved_tus == 0:
        return (
            None,
            EXIT_USAGE,
            [
                (
                    f"blind scan-deps graph: the database lists {len(tus)} translation "
                    "units but none resolve into the scanned roots - a degraded run is "
                    "never green (check the database's paths)"
                )
            ],
        )
    # A flattened TU list cannot reveal header edges or header-only cycles.
    try:
        direct = build_graph(contract, repo, files, _Observed(tu_sets, containing))
    except ContractError as error:
        return None, EXIT_USAGE, [f"direct include graph refused: {error}"]
    direct_adj, _, direct_edges, direct_ext, direct_forbidden, unresolved = direct
    for edge, count in direct_edges.items():
        mod_edges[edge] = max(mod_edges.get(edge, 0), count)
    for key, count in direct_ext.items():
        ext_counts[key] = max(ext_counts.get(key, 0), count)
    forbidden_hits += direct_forbidden
    return (
        (direct_adj, file_mod, mod_edges, ext_counts, forbidden_hits, unresolved),
        0,
        [],
    )


def check_nesting(contract):
    """THE NESTING LAW: a module must not depend on its own child module -
    the deeper namespace uses its shallower ancestors, never the reverse
    (foo must not include foo/bar)."""
    out = []
    for a, b in sorted(contract.edges):
        if b.startswith(a + "/"):
            out.append(
                f"NESTING LAW: {a} must not depend on its child module {b} - "
                "the deeper module uses its ancestors, never the reverse"
            )
    return out


def compute_smells(contract, files, file_mod, mod_edges, file_lines):
    """The refactoring-catalog proxies (refactoring.guru), deterministic over
    the measured module graph. -> [{"kind", "subject", "note"}]: queue items
    for the review cadence, NEVER findings - the thresholds are judgment,
    so a smell can never fail a run. Module-level proxies, honestly named:
    the catalog defines its smells at class/method level.

    kind: library relaxes the FAN-IN-based smells (orphan, island) and the
    module-VOLUME smell (large module): for a library, zero in-repo
    consumers is the definition of public API, and total volume in one
    namespace is a fact (the STL argument), not dead weight. The per-FILE
    volume rule is NOT here - it is a standing rule of its own
    (check_header_size), valid in both kinds."""
    lim = contract.smell_limits
    targets: dict[str, set[str]] = {}
    sources: dict[str, set[str]] = {}
    for a, b in mod_edges:
        targets.setdefault(a, set()).add(b)
        sources.setdefault(b, set()).add(a)
    mods = sorted({m for m in file_mod.values() if m is not None})
    smells: list[dict[str, str]] = []
    done: set[str] = set()
    is_library = contract.kind == "library"
    # exempt = the topmost occupied layer (entry points hang there by design);
    # in library mode EVERY module is exempt from fan-in-based smells - zero
    # in-repo consumers is what public API means
    top_layer = None
    for lay in reversed(contract.stack):
        if any(contract.layer_of.get(m) == lay for m in mods):
            top_layer = lay
            break
    for m in mods:
        fin, fout = sources.get(m, set()), targets.get(m, set())
        if not fin and not fout:
            if is_library:
                continue
            smells.append(
                {
                    "kind": "island module",
                    "subject": m,
                    "note": "no consumers, no dependencies "
                    "(Speculative Generality / Dead Code proxy)",
                }
            )
            done.add(m)
        elif (
            not fin
            and not is_library
            and top_layer
            and contract.layer_of.get(m) != top_layer
        ):
            smells.append(
                {
                    "kind": "orphan module",
                    "subject": m,
                    "note": "no consumer (Lazy Class / Dead Code proxy)",
                }
            )
            done.add(m)
    for m in mods:
        if m in done:
            continue
        fin, fout = sources.get(m, set()), targets.get(m, set())
        if len(fin) == 1 and len(fout) == 1:
            s, t = next(iter(fin)), next(iter(fout))
            in_w = sum(n for (a, b), n in mod_edges.items() if a == s and b == m)
            out_w = sum(n for (a, b), n in mod_edges.items() if a == m and b == t)
            if s != t and in_w >= lim["mutual"] and out_w >= lim["mutual"]:
                smells.append(
                    {
                        "kind": "pass-through module",
                        "subject": m,
                        "note": f"all of it flows {s} -> {m} -> {t} (Middle Man proxy)",
                    }
                )
    for a, b in sorted(mod_edges):
        if (
            a < b
            and mod_edges[(a, b)] >= lim["mutual"]
            and mod_edges.get((b, a), 0) >= lim["mutual"]
        ):
            smells.append(
                {
                    "kind": "mutual pair",
                    "subject": f"{a} <-> {b}",
                    "note": f"x{mod_edges[(a, b)]} each way "
                    "(Inappropriate Intimacy proxy)",
                }
            )
    for m in mods:
        fi, fo = len(sources.get(m, set())), len(targets.get(m, set()))
        if fi >= lim["fan"] or fo >= lim["fan"]:
            smells.append(
                {
                    "kind": "hub module",
                    "subject": m,
                    "note": f"fan-in {fi}, fan-out {fo} "
                    "(Shotgun Surgery / Divergent Change proxy)",
                }
            )
    if not is_library:
        # large MODULE, application only: total volume in one module is
        # bloat; for a library it is a fact (the STL argument). The per-FILE
        # rule is a standing rule of its own (check_header_size), valid in
        # both kinds.
        mod_files: dict[str, int] = {}
        mod_lines: dict[str, int] = {}
        for p in files:
            m = file_mod.get(p)
            if m is None:
                continue
            mod_files[m] = mod_files.get(m, 0) + 1
            mod_lines[m] = mod_lines.get(m, 0) + file_lines.get(p, 0)
        for m in mods:
            files_n, lines_n = mod_files.get(m, 0), mod_lines.get(m, 0)
            if files_n >= lim["files"] or lines_n >= lim["lines"]:
                smells.append(
                    {
                        "kind": "large module",
                        "subject": m,
                        "note": f"{files_n} files, {lines_n} lines (Large Class proxy)",
                    }
                )
    return smells


def check_header_size(contract, files, file_lines):
    """THE HEADER-SIZE RULE - a standing rule, valid in BOTH kinds, never
    switchable off: bloat is not good anywhere. A single file over the
    lines limit is reported wherever it lives; the tier (warn/error) is
    the repo's ratchet decision."""
    out = []
    for p in sorted(files, key=lambda k: files[k]):
        if file_lines.get(p, 0) >= contract.smell_limits["lines"]:
            out.append(
                f"HEADER SIZE: {files[p]} ({file_lines[p]} lines; "
                f"limit {contract.smell_limits['lines']})"
            )
    return out


def check_edges(contract, files, file_mod, mod_edges):
    """DIRECTION (default deny) + CLASSIFICATION over the measured edges.
    -> (violations, warnings): an UNDECLARED edge between two modules of the
    SAME layer obeys `within_layer` (deny | allow | warn); everything else
    is a violation. Warnings never fail the run."""
    violations = []
    warnings = []
    if contract.unclassified == "error":
        for p in sorted(files, key=lambda k: files[k]):
            if file_mod[p] is None:
                violations.append(
                    f"unclassified module: {files[p]} (matches no declared module)"
                )
    for (a, b), n in sorted(mod_edges.items()):
        if a is None or b is None:
            continue  # unclassified sides are flagged per policy above
        if (a, b) in contract.edges:
            continue
        la, lb = contract.layer_of.get(a), contract.layer_of.get(b)
        if la and la == lb and contract.within_layer in ("allow", "warn"):
            if contract.within_layer == "warn":
                warnings.append(
                    f"{a} must not include {b} (x{n}; edge not declared, "
                    f"within layer {la})"
                )
            continue
        violations.append(f"{a} must not include {b} (x{n}; edge not declared)")
    return violations, warnings


def check_stack(contract):
    """THE LAYER STACK: even a DECLARED edge may not run against the stack
    (a higher layer may not be consumed from below it)."""
    out = []
    for a, b in sorted(contract.edges):
        la, lb = contract.layer_of.get(a), contract.layer_of.get(b)
        if la and lb and contract.rank[lb] > contract.rank[la]:
            out.append(
                f"{a} (layer {la}) must not depend on {b} (layer {lb}) - "
                "the layer stack flows one way"
            )
    return out


def check_cycles(contract, files, adj):
    out = []
    for comp in scc(list(files), adj):
        if len(comp) > 1:
            names = sorted(files[x] for x in comp)
            out.append("include cycle: " + " <-> ".join(names))
    return out


def check_namespaces(contract, files, file_mod):
    ns_re = re.compile(
        r"^\s*namespace\s+("
        + re.escape(contract.ns_root)
        + r"(?:::[A-Za-z_]\w*)*)\s*\{",
        re.MULTILINE,
    )
    out = []
    for p in sorted(files, key=lambda k: files[k]):
        mod = file_mod[p]
        if mod is None:
            continue
        if mod == contract.ns_module:
            expected = contract.ns_root
        elif mod.startswith(contract.ns_module + "/"):
            expected = (
                contract.ns_root
                + "::"
                + mod[len(contract.ns_module) + 1 :].replace("/", "::")
            )
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
    span = _diagram_span(text)
    if span is None:
        return []  # no marked diagram = visualization not maintained by the tool
    a, b = span
    shown = text[a + len(DIAGRAM_BEGIN) : b].strip().replace("\r\n", "\n")
    want = contract.marked_diagram().strip()
    if shown != want:
        return [
            (
                "DIAGRAM DRIFT: the diagram section is stale "
                "(run check_arch.py --diagram)"
            )
        ]
    return []


def write_diagram(doc_path, contract, text):
    span = _diagram_span(text)
    if span is None:
        return False
    a, b = span
    section = contract.marked_diagram()
    # keep the document's own line-ending habit: a generated LF-only section
    # inside a CRLF working tree would leave the file mixed (the joiners too)
    nl = (
        "\r\n" if text.count("\r\n") > (text.count("\n") - text.count("\r\n")) else "\n"
    )
    if nl != "\n":
        section = section.replace("\n", "\r\n")
    text = text[: a + len(DIAGRAM_BEGIN)] + nl + section + nl + text[b:]
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
    """-> (files, quarantined): the derive scan prunes the excludes and the
    vendored / non-source names (VENDOR_DIRS); `quarantined` lists the
    vendored names it met, so the draft's `exclude:` prunes them as well."""
    files, pruned = _walk_sources(
        repo, roots, extensions, [*excludes, *sorted(VENDOR_DIRS)]
    )
    quarantined = sorted({rel.rsplit("/", 1)[-1] for rel in pruned} & VENDOR_DIRS)
    return files, quarantined


def topo_leaf_first(units, mod_edges):
    """(order, cyclic leftover): dependencies first (Kahn, alpha tie-break)."""
    deps: dict[str, set[str]] = {u: set() for u in units}
    for a, b in mod_edges:
        deps.setdefault(a, set()).add(b)
    order: list[str] = []
    remaining = set(units)
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
    dependencies as they are -> a draft ARCHITECTURE.md. The draft resolves
    includes with the check's own resolver and excludes what the scan
    pruned, so the unedited draft passes the check on the same tree
    (cycles aside - they are reported for fixing first)."""
    roots = [_norm_root(r) for r in roots]
    files, quarantined = derive_scan(repo, roots, extensions, excludes)
    warnings: list[str] = []
    if not files:
        return [], [f"no sources under {roots} - nothing to derive"]
    exclude = list(dict.fromkeys([*excludes, *quarantined]))
    key_of: dict[str, str] = {}
    outside = 0
    for p, rel in files.items():
        root = next((r for r in roots if r == "." or rel.startswith(r + "/")), None)
        if root is None:
            outside += 1
            continue
        sub = posixpath.dirname(rel if root == "." else rel[len(root) + 1 :])
        if sub:
            key_of[p] = sub
        elif root != ".":
            key_of[p] = root  # a root-level file: the root itself is the module
        else:
            outside += 1  # a file directly in the repository root
    if outside:
        warnings.append(
            f"{outside} file(s) sit outside every module directory - not derived"
        )
    units = sorted(set(key_of.values()))
    # first segments that hold scanned sources are project directories - an
    # include into one is never external, even when it does not resolve
    project_dirs = {k.split("/")[0] for k in units}
    resolver = Contract("```arch\nscan: " + ", ".join(roots) + "\n```\n", "derive")
    mod_edges = set()
    ext = set()
    unresolved = 0
    unresolved_samples: list[str] = []
    for p, rel in files.items():
        a = key_of.get(p)
        if a is None:
            continue
        with open(p, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        for m in INCLUDE_RE.finditer(text):
            kind, inc = m.group(1), m.group(2)
            t = resolve(resolver, repo, inc, kind, p, files)
            if t is None:
                first = inc.split("/")[0]
                if (
                    "/" in inc
                    and not inc.startswith(".")
                    and first not in roots
                    and first not in project_dirs
                ):
                    ext.add(first)
                elif kind == '"':
                    unresolved += 1
                    if len(unresolved_samples) < 5:
                        unresolved_samples.append(f'{rel}: "{inc}"')
                continue
            b = key_of.get(t)
            if b is not None and a != b:
                mod_edges.add((a, b))
    # the layer proposal: one layer per first path segment, stacked by the
    # measured segment-level dependency order (bottom-up)
    segs = sorted({u.split("/")[0] for u in units})
    seg_edges = sorted(
        {
            (a.split("/")[0], b.split("/")[0])
            for a, b in mod_edges
            if a.split("/")[0] != b.split("/")[0]
        }
    )
    stack, cyclic_layers = topo_leaf_first(segs, seg_edges)
    full_stack = stack + cyclic_layers
    if cyclic_layers:
        warnings.append(
            "layer cycle among "
            + ", ".join(cyclic_layers)
            + " - fix the cycle first; the stack stops at the cycle"
        )
    order, cyclic = topo_leaf_first(units, sorted(mod_edges))
    if cyclic:
        warnings.append(
            "cycle among "
            + ", ".join(cyclic)
            + " - fix the cycle first; the edges are still emitted, "
            "the order stops at the cycle"
        )

    arch = []
    knobs = [
        ("scan", ", ".join(roots)),
        ("extensions", ", ".join(extensions)),
        ("exclude", ", ".join(exclude)),
        ("unclassified", "error"),
    ]
    if ext:
        knobs.append(("external", ", ".join(sorted(ext))))
    knobs += [
        ("forbidden", ""),
        ("quotes_relative", "true"),
        ("repo_fallback", "false"),
        ("cycles", "true"),
        ("namespace_root", ""),
        ("namespace_module", ""),
        ("namespace_require", ""),
    ]
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

    out = [
        "# Architecture",
        "",
        "<!-- GENERATED by check_arch.py --derive - the MEASURED architecture.",
        "     REVIEW AND TIGHTEN: every `allow` edge below is what the code does",
        "     TODAY; delete the edges that should not exist and reshape the",
        "     layer stack, then keep this file as the checked-in contract.",
        "     ALSO CONSIDER: scope `scan:` to the shipped sources - test and tool",
        "     trees are consumers by construction and usually stay out of the",
        "     contract; prune the `external:` list (system headers leak in as",
        "     noise, e.g. <sys/stat.h>-style prefixes). -->",
        "",
        "## The machine contract",
        "",
        "```arch",
    ]
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
    out.append(
        Contract(
            "# Architecture\n\n```arch\n" + "\n".join(arch) + "\n```\n", "draft"
        ).marked_diagram()
    )
    out.append(DIAGRAM_END)
    out.append("")
    if unresolved:
        warnings.append(
            f"{unresolved} quoted include(s) did not resolve - re-check "
            "after adoption; first: " + " | ".join(unresolved_samples)
        )
    return out, warnings


# ------------------------------------------------------------------ core ---
def collect(repo, doc_path, report=False):
    """The CHECK step: the code against the contract.
    -> (exit code, human lines, machine summary)."""
    resolved = None
    try:
        resolved, text = load_contract_text(repo, doc_path)
        contract = Contract(text, resolved)
        _diagram_span(text)  # malformed diagram markers refuse the run
        files = scan_files(contract, repo)
    except ContractError as e:
        return (
            EXIT_USAGE,
            [f"contract error: {e}"],
            {
                "schema": 1,
                "code": EXIT_USAGE,
                "doc": resolved,
                "error": str(e),
                "violations": None,
            },
        )
    if not files:
        msg = f"skip: no sources under {contract.roots} (contract {resolved})"
        return (
            EXIT_SKIP,
            [msg],
            {
                "schema": 1,
                "code": EXIT_SKIP,
                "doc": resolved,
                "files": 0,
                "violations": None,
                "error": msg,
            },
        )
    if contract.graph == "scan-deps":
        graph, gcode, glines = build_graph_scan_deps(contract, repo, files)
        if graph is None:
            return (
                gcode,
                glines,
                {
                    "schema": 1,
                    "code": gcode,
                    "doc": resolved,
                    "files": len(files),
                    # only an incomplete scan is a finding; a refusal or a
                    # skip never evaluated the rules
                    "violations": glines if gcode == EXIT_VIOLATIONS else None,
                    "error": glines[0],
                },
            )
        adj, file_mod, mod_edges, ext_counts, forbidden_hits, unresolved = graph
    else:
        adj, file_mod, mod_edges, ext_counts, forbidden_hits, unresolved = build_graph(
            contract, repo, files
        )
    violations = list(forbidden_hits)
    violations += check_stack(contract)
    edge_violations, warnings = check_edges(contract, files, file_mod, mod_edges)
    violations += edge_violations
    if contract.nesting == "error":
        violations += check_nesting(contract)
    if contract.cycles:
        violations += check_cycles(contract, files, adj)
    if contract.ns_root:
        violations += check_namespaces(contract, files, file_mod)
    if contract.unresolved_policy == "error" and unresolved:
        violations += [f"unresolved include: {u}" for u in unresolved]
    violations += check_diagram(contract, text)
    smells: list[dict[str, str]] = []
    # the header-size rule is standing: line counts are always measured
    file_lines: dict[str, int] = {}
    for p in files:
        with open(p, encoding="utf-8", errors="replace") as fh:
            file_lines[p] = sum(1 for _ in fh)
    if contract.smells != "off":
        smells = compute_smells(contract, files, file_mod, mod_edges, file_lines)
        if contract.smells == "warn":
            warnings += [
                f"SMELL {s['kind']}: {s['subject']} ({s['note']})" for s in smells
            ]
    header_findings = check_header_size(contract, files, file_lines)
    if contract.header_size == "warn":
        warnings += header_findings
    elif contract.header_size == "error":
        violations += header_findings

    summary = {
        "schema": 1,
        "code": None,
        "doc": resolved,
        "graph": contract.graph,
        "files": len(files),
        "modules": sorted(contract.modules),
        "layer_stack": list(contract.stack),
        "declared_edges": [list(e) for e in sorted(contract.edges)],
        "measured_edges": [[a, b, n] for (a, b), n in sorted(mod_edges.items())],
        "external": ext_counts,
        "violations": violations,
        "warnings": warnings,
        "smells": smells,
        "large_headers": header_findings,
        "clean": not violations,
    }
    lines = []
    if report:
        if contract.stack:
            lines.append("layer stack (bottom-up): " + " <- ".join(contract.stack))
            for lay in contract.stack:
                members = sorted(
                    m for m in contract.modules if contract.layer_of.get(m) == lay
                )
                if members:
                    lines.append(f"  {lay}: " + ", ".join(members))
            floating = sorted(m for m in contract.modules if m not in contract.layer_of)
            if floating:
                lines.append("  (unlayered): " + ", ".join(floating))
        lines.append(
            "declared edges: "
            + ", ".join(f"{a}->{b}" for a, b in sorted(contract.edges))
        )
        lines.append("measured module edges:")
        for (a, b), n in sorted(mod_edges.items()):
            ok = "" if (a, b) in contract.edges else "  NOT DECLARED"
            lines.append(f"  {a} -> {b}  x{n}{ok}")
        if ext_counts:
            lines.append(
                "external (allowed, reported): "
                + ", ".join(f"{k} x{v}" for k, v in sorted(ext_counts.items()))
            )
        if unresolved and contract.unresolved_policy != "error":
            lines.append("unresolved quoted includes (info only):")
            for u in unresolved[:10]:
                lines.append("  " + u)
        if contract.smells != "off":
            lines.append(
                f"refactoring smells ({len(smells)}; report-only, never findings):"
            )
            for s in smells:
                lines.append(f"  {s['kind']}: {s['subject']} ({s['note']})")
        if contract.header_size == "report" and header_findings:
            lines.append(f"large files ({len(header_findings)}; standing rule):")
            for h in header_findings:
                lines.append("  " + h)
    for w in warnings:
        lines.append("warning: " + w)
    if violations:
        summary["code"] = EXIT_VIOLATIONS
        return (
            EXIT_VIOLATIONS,
            lines
            + [f"{len(violations)} architecture violation(s):"]
            + ["  " + v for v in violations],
            summary,
        )
    # name only what this run actually checked
    checked = [f"{len(files)} files", f"{len(mod_edges)} distinct module edges"]
    if contract.cycles:
        checked.append("no cycles")
    if contract.stack:
        checked.append("layer stack holds")
    if contract.ns_root:
        checked.append("namespaces mapped")
    lines.append(f"architecture clean ({', '.join(checked)})")
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
        return (
            "# Architecture\n\n"
            + DIAGRAM_BEGIN
            + "\n"
            + c.marked_diagram()
            + "\n"
            + DIAGRAM_END
            + "\n\n```arch\n"
            + body.strip()
            + "\n```\n"
        )

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
        print(
            f"selftest {name}: {'PASS' if good else f'FAIL (got {got}, want {want})'}"
        )

    KNOBS = "scan: src\n"
    # 1 declared edges are legal; the chain passes
    b1 = (
        KNOBS
        + """
module base
module mid
module top
allow mid -> base
allow top -> mid
"""
    )
    files1 = {
        "src/base/a.h": "#pragma once\n",
        "src/mid/m.h": '#include "base/a.h"\n',
        "src/top/t.h": '#include "mid/m.h"\n',
    }
    e, out = run(files1, contract(b1))
    expect("declared chain clean", (e, [v for v in out if "must" in v]), (0, []))

    # 2 default deny: an undeclared edge fires
    b2 = KNOBS + "module base\nmodule top\n"
    files2 = {"src/base/a.h": "#pragma once\n", "src/top/t.h": '#include "base/a.h"\n'}
    e, out = run(files2, contract(b2))
    expect(
        "undeclared edge fires (default deny)",
        (e, any("top must not include base" in v for v in out)),
        (1, True),
    )

    # 3 longest-prefix module mapping + upward refusal
    b3 = KNOBS + "module proj\nmodule proj/sub\nallow proj/sub -> proj\n"
    files3 = {
        "src/proj/root.h": '#include "proj/sub/s.h"\n',
        "src/proj/sub/s.h": "#pragma once\n",
    }
    e, out = run(files3, contract(b3))
    expect(
        "longest-prefix mapping + upward refusal",
        (e, any("proj must not include proj/sub" in v for v in out)),
        (1, True),
    )

    # 4 the layer stack: downward declared edges hold, upward ones violate
    b4 = (
        KNOBS + "layer base\nlayer mid\nlayer top\n"
        "module base/x in base\nmodule mid/y in mid\nmodule top/z in top\n"
        "allow top/z -> mid/y\nallow mid/y -> base/x\n"
    )
    files4 = {
        "src/base/x/x.h": "#pragma once\n",
        "src/mid/y/y.h": '#include "base/x/x.h"\n',
        "src/top/z/z.h": '#include "mid/y/y.h"\n',
    }
    e, out = run(files4, contract(b4))
    expect(
        "stacked layers: downward edges hold",
        (e, [v for v in out if "layer" in v and "must" in v]),
        (0, []),
    )
    b4bad = (
        KNOBS + "layer base\nlayer mid\nlayer top\n"
        "module base/x in base\nmodule mid/y in mid\nmodule top/z in top\n"
        "allow mid/y -> base/x\nallow base/x -> mid/y\n"
    )
    e, out = run(files4, contract(b4bad))
    expect(
        "stacked layers: an UPWARD declared edge violates the stack",
        (
            e,
            any(
                "base/x (layer base) must not depend on mid/y (layer mid)" in v
                for v in out
            ),
        ),
        (1, True),
    )
    b4typo = KNOBS + "layer base\nmodule x in middle\n"
    e, out = run({}, contract(b4typo))
    expect(
        "layer typo = contract error",
        (e, any("undeclared layer" in v for v in out)),
        (2, True),
    )

    # 5 unclassified policy
    b5 = KNOBS + "module known\n"
    e, out = run(
        {"src/known/k.h": "#pragma once\n", "src/stray/s.h": "#pragma once\n"},
        contract(b5 + "unclassified: error\n"),
    )
    expect(
        "unclassified=error flags the stray file",
        (e, any("unclassified module: src/stray/s.h" in v for v in out)),
        (1, True),
    )
    e, out = run(
        {"src/known/k.h": "#pragma once\n", "src/stray/s.h": "#pragma once\n"},
        contract(b5 + "unclassified: ignore\n"),
    )
    expect(
        "unclassified=ignore stays silent",
        (e, [v for v in out if "unclassified" in v]),
        (0, []),
    )

    # 6 cycles: file-level SCC
    e, out = run(
        {
            "src/known/a.h": '#include "known/b.h"\n',
            "src/known/b.h": '#include "known/a.h"\n',
        },
        contract(b5),
    )
    expect(
        "cycle detected",
        (e, any("include cycle:" in v and "a.h" in v for v in out)),
        (1, True),
    )

    # 7 resolution: rooted via scan roots, quoted-relative, repo fallback
    b7 = (
        "scan: src, include\nrepo_fallback: true\n"
        "module proj\nmodule proj/sub\nallow proj -> proj/sub\n"
    )
    files7 = {
        "src/proj/rf.h": '#include "proj/sub/s.h"\n',
        "include/proj/rf2.h": '#include "proj/sub/s.h"\n',
        "src/proj/sub/s.h": "#pragma once\n",
    }
    e, out = run(files7, contract(b7))
    expect(
        "resolution: rooted via scan roots + repo fallback",
        (e, [v for v in out if "must" in v]),
        (0, []),
    )

    # 8 external reported, forbidden violates
    b8 = "scan: src\nexternal: other\nforbidden: renderer\nmodule lib\n"
    e, out = run(
        {
            "src/lib/l.h": '#include <other/x.h>\n#include "renderer/r.h"\n'
            "#include <system.h>\n"
        },
        contract(b8),
    )
    expect(
        "forbidden fires, external allowed",
        (e, any('must not include "renderer/r.h"' in v for v in out)),
        (1, True),
    )

    # 9 the namespace gate
    b9 = (
        KNOBS
        + "namespace_root: proj\nnamespace_module: proj\nnamespace_require: .hpp\n"
        "module proj\nmodule proj/sub\nallow proj -> proj/sub\n"
    )
    ns_files = {
        "src/proj/Root.hpp": "namespace proj {\nint f();\n}\n",
        "src/proj/sub/S.hpp": "namespace proj::sub::Deep {\nint g();\n}\n",
        "src/proj/impl.cpp": "namespace proj::sub {\nint g() { return 0; }\n}\n",
        "src/proj/plain.cpp": "int helper() { return 1; }\n",
    }
    e, out = run(ns_files, contract(b9))
    expect(
        "namespace gate: own namespaces pass",
        (e, [v for v in out if "never opens" in v]),
        (0, []),
    )
    ns_bad = dict(ns_files)
    ns_bad["src/proj/sub/S.hpp"] = "int g();\n"
    e, out = run(ns_bad, contract(b9))
    expect(
        "namespace gate: header without its namespace fires",
        (
            e,
            any(
                "src/proj/sub/S.hpp: never opens namespace proj::sub" in v for v in out
            ),
        ),
        (1, True),
    )
    ns_bad2 = dict(ns_files)
    ns_bad2["src/proj/sub/impl.cpp"] = (
        "namespace proj::other {\nint g() { return 0; }\n}\n"
    )
    e, out = run(ns_bad2, contract(b9))
    expect(
        "namespace gate: cpp opening a foreign namespace fires",
        (
            e,
            any(
                "src/proj/sub/impl.cpp: never opens namespace proj::sub" in v
                for v in out
            ),
        ),
        (1, True),
    )
    # the forward-declaration split: a header in a foreign (non-root) namespace
    # still fires; a cpp opening ONLY a fully foreign namespace is exempt
    ns_bad3 = dict(ns_files)
    ns_bad3["src/proj/sub/S.hpp"] = "namespace completely_foreign {\nint g();\n}\n"
    e, out = run(ns_bad3, contract(b9))
    expect(
        "namespace gate: header in a foreign namespace still fires",
        (
            e,
            any(
                "src/proj/sub/S.hpp: never opens namespace proj::sub" in v for v in out
            ),
        ),
        (1, True),
    )
    ns_ok2 = dict(ns_files)
    ns_ok2["src/proj/plain2.cpp"] = (
        "namespace completely_foreign {\nint h() { return 2; }\n}\n"
    )
    e, out = run(ns_ok2, contract(b9))
    expect(
        "namespace gate: cpp in a fully foreign namespace only is exempt",
        (e, [v for v in out if "never opens" in v]),
        (0, []),
    )

    # 10 the diagram drift gate + the block layout shape
    e, out = run(files4, diagram_of(b4))
    expect(
        "fresh generated diagram passes the drift gate",
        (e, [v for v in out if "DRIFT" in v]),
        (0, []),
    )
    stale = diagram_of(b4).replace("block-beta", "block-beta\n    %% stale")
    e, out = run(files4, stale)
    expect(
        "stale diagram fires", (e, any("DIAGRAM DRIFT" in v for v in out)), (1, True)
    )
    d = Contract(contract(b4), "doc").marked_diagram()
    expect(
        "diagrams: the drill-down (stack + per layer), pull spans, slack",
        "### the layers" in d
        and 'top["top"]:2' in d
        and 'mid["mid"]:2' in d
        and 'base["base"]' in d
        and "space" in d
        and "### layer base - the modules" in d
        and 'base_x["base/x"]' in d
        and 'top_z["top/z"]:2' in d,
        True,
    )

    # 10b the drill-down into nested containers
    b10 = (
        KNOBS + "layer base\nmodule proj in base\nmodule proj/sub in base\n"
        "allow proj/sub -> proj\n"
    )
    d10 = Contract(contract(b10), "doc").marked_diagram()
    expect(
        "drill-down: container module resolves into its own diagram",
        "### layer base - the modules" in d10
        and "### proj - the nested modules" in d10
        and 'proj_sub["proj/sub"]' in d10,
        True,
    )

    # 11 contract errors are loud (exit 2)
    e, out = run(files1, contract(b1 + "allow mid -> typo\n"))
    expect(
        "edge to an undeclared module = contract error",
        (e, any("contract error" in v for v in out)),
        (2, True),
    )
    e, out = run(files1, "# Architecture\n\nno arch block here\n")
    expect(
        "missing arch block = contract error",
        (e, any("no ```arch block" in v for v in out)),
        (2, True),
    )

    # 12 derive: measured contract + layer proposal; a cycle is reported
    with tempfile.TemporaryDirectory(prefix="check_arch_deriv_") as td:
        write(
            td,
            {
                "src/leaf/l.h": "#pragma once\n",
                "src/mid/m.h": '#include "leaf/l.h"\n',
                "src/top/t.h": '#include "mid/m.h"\n',
            },
        )
        lines, warns = derive(td, ["src"], [".h"], DEFAULT_EXCLUDES)
        text = "\n".join(lines)
        expect(
            "derive: layers + modules + edges in the draft",
            "layer leaf" in text
            and "module mid in mid" in text
            and "allow mid -> leaf" in text
            and "allow top -> mid" in text,
            True,
        )
        expect("derive: no warnings on a clean DAG", warns, [])
        write(td, {"src/leaf/l.h": '#include "top/t.h"\n'})
        lines, warns = derive(td, ["src"], [".h"], DEFAULT_EXCLUDES)
        expect("derive: cycle reported", any("cycle" in w for w in warns), True)

    # 12b the quoted resolution chains (the F1 chains)
    b12 = (
        "scan: src\nquotes_resolve: root\n"
        "module proj\nmodule proj/sub\nallow proj -> proj/sub\n"
    )
    files12 = {
        "src/proj/sub/s.h": "#pragma once\n",
        "src/proj/r.h": '#include "src/proj/sub/s.h"\n',
    }
    e, out = run(files12, contract(b12))
    expect(
        "quotes_resolve: root-relative chain",
        (e, [v for v in out if "must" in v]),
        (0, []),
    )
    b12m = (
        "scan: src\nquotes_resolve: includer-then-root\n"
        "module proj\nmodule proj/sub\nallow proj -> proj/sub\n"
    )
    files12m = {
        "src/proj/sub/s.h": "#pragma once\n",
        "src/proj/a.h": '#include "sub/s.h"\n',
        "src/proj/b.h": '#include "src/proj/sub/s.h"\n',
    }
    e, out = run(files12m, contract(b12m))
    expect(
        "quotes_resolve: the mixed includer-then-root chain",
        (e, [v for v in out if "must" in v]),
        (0, []),
    )

    # 12c within_layer: the same-layer policy for undeclared sibling edges
    b12c = "scan: src\nlayer same\nmodule same/a in same\nmodule same/b in same\n"
    files12c = {
        "src/same/a/x.h": "#pragma once\n",
        "src/same/b/y.h": '#include "same/a/x.h"\n',
    }
    e, out = run(files12c, contract(b12c))
    expect(
        "within_layer deny (default): sideways edge fires",
        (e, any("same/b must not include same/a" in v for v in out)),
        (1, True),
    )
    e, out = run(files12c, contract(b12c + "within_layer: warn\n"))
    expect(
        "within_layer warn: reported, not failing",
        (e, any("within layer same" in v for v in out)),
        (0, True),
    )
    e, out = run(files12c, contract(b12c + "within_layer: allow\n"))
    expect(
        "within_layer allow: silent",
        (e, [v for v in out if "must not include" in v]),
        (0, []),
    )

    # 12d derive quarantines vendor dirs (F2)
    with tempfile.TemporaryDirectory(prefix="check_arch_vendor_") as td:
        write(
            td,
            {
                "src/mid/m.h": "#pragma once\n",
                "third_party/imgui/i.h": "#pragma once\n",
            },
        )
        expect("auto-detect skips vendor dirs", detect_roots(td), ["src"])

    # 12e a CRLF document keeps its habit through --diagram (A3)
    with tempfile.TemporaryDirectory(prefix="check_arch_crlf_") as td:
        doc_text = diagram_of(b5).replace("\n", "\r\n")
        p = os.path.join(td, "ARCHITECTURE.md")
        with open(p, "w", encoding="utf-8", newline="") as fh:
            fh.write(doc_text)
        c5 = Contract(contract(b5), "doc")
        expect(
            "--diagram rewrites the marked section",
            write_diagram(p, c5, doc_text),
            True,
        )
        with open(p, "rb") as bfh:
            data = bfh.read()
        expect(
            "CRLF doc stays CRLF through --diagram",
            data.count(b"\r\n") > 10 and data.count(b"\n") == data.count(b"\r\n"),
            True,
        )
        with open(p, encoding="utf-8", newline="") as fh:
            reread = fh.read()
        expect(
            "CRLF doc passes the drift gate after the rewrite",
            check_diagram(c5, reread) == [],
            True,
        )

    # 12f the scan-deps operational contract (A1/A2): a blind graph is loud,
    # a missing tool is a skip - proven with a fake clang-scan-deps
    if os.name == "nt":
        outer = tempfile.mkdtemp(prefix="check_arch_sd_")
        try:
            other = os.path.join(outer, "elsewhere")
            os.makedirs(other)
            tu = os.path.join(other, "x.cpp")
            with open(tu, "w") as fh:
                fh.write("int x;\n")
            fake = os.path.join(outer, "fake-csd.cmd")
            # the make format escapes blanks; a TEMP path may contain them
            mk = tu.replace(" ", "\\ ")
            with open(fake, "w") as fh:
                fh.write("@echo off\r\necho " + mk + ": \\\r\necho   " + mk + "\r\n")
            db_text = json.dumps(
                [{"directory": outer, "command": "cl /c x.cpp", "file": tu}]
            )
            sd = (
                "scan: src\ngraph: scan-deps\ndb: compile_commands.json\n"
                "clang_scan_deps: " + fake + "\nmodule src\n"
            )
            e, out = run(
                {"src/a.cpp": "int x;\n", "compile_commands.json": db_text},
                contract(sd),
            )
            expect(
                "blind scan-deps graph is refused (exit 2), never green",
                (e, any("blind scan-deps graph" in v for v in out)),
                (2, True),
            )
            missing = (
                "scan: src\ngraph: scan-deps\ndb: compile_commands.json\n"
                "clang_scan_deps: "
                + os.path.join(outer, "no-such-tool.exe")
                + "\nmodule src\n"
            )
            e, out = run(
                {"src/a.cpp": "int x;\n", "compile_commands.json": db_text},
                contract(missing),
            )
            expect(
                "missing clang-scan-deps is a skip (exit 77), no traceback",
                (e, any("clang-scan-deps not found" in v for v in out)),
                (77, True),
            )
        finally:
            shutil.rmtree(outer, ignore_errors=True)

    # Direct header relationships remain visible behind a flattened compiler inventory.
    from unittest.mock import patch

    def scan_deps_control(cycle=False, declare=False, scanner_code=0, resource=False):
        with tempfile.TemporaryDirectory(prefix="check_arch_headers_") as td:
            tu = os.path.join(td, "src", "app", "main.cpp")
            first = os.path.join(td, "src", "shared", "first.h")
            last = os.path.join(td, "src", "model", "last.h")
            files = {
                "src/app/main.cpp": '#include "shared/first.h"\n',
                "src/shared/first.h": '#pragma once\n#include "model/last.h"\n',
                "src/model/last.h": "#pragma once\n"
                + ('#include "shared/first.h"\n' if cycle else ""),
                "compile_commands.json": json.dumps(
                    [{"directory": td, "command": "c++ -c main.cpp", "file": tu}]
                ),
            }
            if resource:
                files["src/app/icon.rc"] = "1 ICON icon.ico\n"
                files["compile_commands.json"] = json.dumps(
                    [
                        {"directory": td, "command": "c++ -c main.cpp", "file": tu},
                        {
                            "directory": td,
                            "command": "rc /fo icon.res icon.rc",
                            "file": os.path.join(td, "src", "app", "icon.rc"),
                        },
                    ]
                )
            body = (
                "scan: src\ngraph: scan-deps\ndb: compile_commands.json\n"
                "clang_scan_deps: synthetic-scanner\n"
                "module src/app\nmodule src/shared\nmodule src/model\n"
                "allow src/app -> src/shared\nallow src/app -> src/model\n"
            )
            if declare:
                body += "allow src/shared -> src/model\n"
            if cycle:
                body += "allow src/model -> src/shared\n"
            write(td, files)
            document = os.path.join(td, "ARCHITECTURE.md")
            write(td, {"ARCHITECTURE.md": contract(body)})
            mk = [p.replace(" ", "\\ ") for p in (tu, first, last)]
            stdout = mk[0] + ": \\\n  " + " \\\n  ".join(mk) + "\n"
            response = subprocess.CompletedProcess(
                [], scanner_code, stdout, "scanner error"
            )

            def scanner(argv, **_kwargs):
                database = argv[1].split("=", 1)[1]
                with open(database, encoding="utf-8") as fh:
                    commands = json.load(fh)
                if any(command["file"].endswith(".rc") for command in commands):
                    return subprocess.CompletedProcess(
                        [], 1, stdout, "error: resource command is not C/C++"
                    )
                return response

            with (
                patch.dict(os.environ, {"TEMP": td}),
                patch("subprocess.run", side_effect=scanner),
            ):
                return run_check(td, document)

    e, out = scan_deps_control()
    expect(
        "scan-deps refuses undeclared header-module edge",
        (e, any("src/shared must not include src/model" in v for v in out)),
        (1, True),
    )
    e, out = scan_deps_control(cycle=True, declare=True)
    expect(
        "scan-deps detects a guarded header-only cycle",
        (e, any("include cycle:" in v for v in out)),
        (1, True),
    )
    e, out = scan_deps_control(declare=True)
    expect("scan-deps accepts the declared acyclic alternative", e, 0)
    e, out = scan_deps_control(declare=True, resource=True)
    expect("scan-deps excludes resource commands without dropping C/C++ coverage", e, 0)
    e, out = scan_deps_control(declare=True, scanner_code=1)
    expect(
        "scan-deps refuses a failed scanner with complete-looking stdout",
        (e, any("clang-scan-deps failed" in v for v in out)),
        (2, True),
    )

    # 12g smells: orphan + island (report tier)
    b12g = (
        "scan: src\nlayer base\nlayer mid\nlayer top\n"
        "module base/a in base\nmodule base/b in base\nmodule base/c in base\n"
        "module mid/m in mid\nmodule top/t in top\n"
        "allow mid/m -> base/a\nallow top/t -> mid/m\n"
        "allow base/b -> base/a\n"
    )
    files12g = {
        "src/base/a/x.h": "#pragma once\n",
        "src/base/b/y.h": '#include "base/a/x.h"\n',
        "src/base/c/z.h": "#pragma once\n",
        "src/mid/m/m.h": '#include "base/a/x.h"\n',
        "src/top/t/t.h": '#include "mid/m/m.h"\n',
    }
    e, out = run(files12g, contract(b12g), report=True)
    joined = "\n".join(out)
    expect(
        "smells: orphan + island detected",
        ("orphan module: base/b" in joined and "island module: base/c" in joined),
        True,
    )

    # 12h smells: pass-through + hub + large (and the smell_limits knob)
    b12h = (
        "scan: src\nlayer same\n"
        "module same/a in same\nmodule same/p in same\nmodule same/b in same\n"
        "module same/root in same\n"
        "module same/s1 in same\nmodule same/s2 in same\n"
        "module same/s3 in same\nmodule same/s4 in same\n"
        "module same/s5 in same\nmodule same/s6 in same\n"
        "module same/s7 in same\nmodule same/s8 in same\n"
        "module same/big in same\n"
        "smell_limits: files=2\n"
        "allow same/b -> same/p\nallow same/p -> same/a\n"
        + "".join(f"allow same/root -> same/s{i}\n" for i in range(1, 9))
    )
    files12h = {
        "src/same/a/a.h": "#pragma once\n",
        "src/same/root/r.h": (
            '#include "same/s1/s1.h"\n'
            '#include "same/s2/s2.h"\n'
            '#include "same/s3/s3.h"\n'
            '#include "same/s4/s4.h"\n'
            '#include "same/s5/s5.h"\n'
            '#include "same/s6/s6.h"\n'
            '#include "same/s7/s7.h"\n'
            '#include "same/s8/s8.h"\n'
        ),
        "src/same/p/p1.h": '#include "same/a/a.h"\n',
        "src/same/p/p2.h": '#include "same/a/a.h"\n',
        "src/same/p/p3.h": '#include "same/a/a.h"\n',
        "src/same/b/b.h": (
            '#include "same/p/p1.h"\n#include "same/p/p2.h"\n#include "same/p/p3.h"\n'
        ),
        "src/same/big/f1.h": "#pragma once\n",
        "src/same/big/f2.h": "#pragma once\n",
    }
    files12h.update({f"src/same/s{i}/s{i}.h": "#pragma once\n" for i in range(1, 9)})
    e, out = run(files12h, contract(b12h), report=True)
    joined = "\n".join(out)
    expect(
        "smells: pass-through + hub + large detected",
        (
            "pass-through module: same/p" in joined
            and "hub module: same/root" in joined
            and "large module: same/big" in joined
        ),
        True,
    )

    # 12i smells: mutual pair; off is silent; warn prints without failing
    b12i = (
        "scan: src\nlayer same\n"
        "module same/p in same\nmodule same/q in same\n"
        "allow same/p -> same/q\nallow same/q -> same/p\n"
    )
    files12i = {
        "src/same/p/p.h": (
            '#include "same/q/q1.h"\n#include "same/q/q2.h"\n#include "same/q/q3.h"\n'
        ),
        "src/same/q/q.h": (
            '#include "same/p/p1.h"\n#include "same/p/p2.h"\n#include "same/p/p3.h"\n'
        ),
        "src/same/q/q1.h": "#pragma once\n",
        "src/same/q/q2.h": "#pragma once\n",
        "src/same/q/q3.h": "#pragma once\n",
        "src/same/p/p1.h": "#pragma once\n",
        "src/same/p/p2.h": "#pragma once\n",
        "src/same/p/p3.h": "#pragma once\n",
    }
    e, out = run(files12i, contract(b12i), report=True)
    expect(
        "smells: mutual pair detected",
        "mutual pair: same/p <-> same/q" in "\n".join(out),
        True,
    )
    e, out = run(files12i, contract(b12i + "smells: off\n"), report=True)
    expect("smells off: silent", "mutual pair" in "\n".join(out), False)
    e, out = run(files12i, contract(b12i + "smells: warn\n"))
    expect(
        "smells warn: SMELL warning lines, exit still 0",
        (e, any(v.startswith("warning: SMELL mutual pair") for v in out)),
        (0, True),
    )

    # 12j kind: library - zero fan-in means public API, not dead code; and
    # volume is judged per FILE, not per module (the STL argument)
    e, out = run(files12g, contract(b12g + "kind: library\n"), report=True)
    expect(
        "kind library: fan-in smells suppressed",
        ("orphan module" in "\n".join(out) or "island module" in "\n".join(out)),
        False,
    )
    e, out = run(files12g, contract(b12g + "kind: plugin\n"))
    expect(
        "kind: invalid value is a contract error",
        (e, any("kind: application|library expected" in v for v in out)),
        (2, True),
    )
    b12j = "scan: src\nkind: library\nsmell_limits: lines=5\nmodule same/big\n"
    files12j = {
        "src/same/big/1.h": "#pragma once\nint a;\nint b;\nint c;\nint d;\nint e;\nint f;\n",
        "src/same/big/2.h": "#pragma once\n",
    }
    e, out = run(files12j, contract(b12j), report=True)
    joined = "\n".join(out)
    expect(
        "kind library: large HEADER (per file) still reported",
        "HEADER SIZE: src/same/big/1.h" in joined,
        True,
    )
    expect("kind library: large module suppressed", "large module" in joined, False)
    e, out = run(files12j, contract(b12j + "smells: off\n"))
    expect(
        "header-size rule: standing, independent of smells",
        any("HEADER SIZE: src/same/big/1.h" in v for v in out),
        True,
    )
    b12k = "scan: src\nsmell_limits: lines=5\nmodule big\n"
    files12k = {
        "src/big/1.h": "#pragma once\nint a;\nint b;\nint c;\nint d;\nint e;\nint f;\n",
        "src/big/2.h": "#pragma once\n",
    }
    e, out = run(files12k, contract(b12k), report=True)
    joined = "\n".join(out)
    expect(
        "application: large module AND large header both reported",
        ("large module: big" in joined and "HEADER SIZE: src/big/1.h" in joined),
        True,
    )
    e, out = run(files12k, contract(b12k + "header_size: error\n"))
    expect(
        "header_size error: the standing rule can be a hard gate",
        (e, any("HEADER SIZE:" in v for v in out)),
        (1, True),
    )

    # 13 exit-code contract
    e, _ = run({"src/known/k.h": "#pragma once\n"}, contract(b5))
    expect("exit 0 on a clean tree", e, 0)
    e, _ = run({"src/stray/s.h": "#pragma once\n"}, contract(b5))
    expect("exit 1 on findings", e, 1)
    e, _ = run({"src/README.txt": "no sources here\n"}, contract(b5))
    expect("exit 77 on nothing to scan (the root exists, no sources)", e, 77)

    # 14 the 1.5.0 pins: a gate never loses sources in silence, resolves the
    # same way on every run, and both graph modes classify alike
    import contextlib
    import io

    # 14a exclude: whole names, globs and repo-relative paths - never a
    # substring, never the checkout location
    with tempfile.TemporaryDirectory(prefix="check_arch_build_") as td:
        repo14 = os.path.join(td, "build", "checkout")
        write(
            repo14,
            {
                "src/core/a.h": "#pragma once\n",
                "src/app/b.h": '#include "core/a.h"\n',
                "ARCHITECTURE.md": contract("scan: src\nmodule core\nmodule app\n"),
            },
        )
        e, out = run_check(repo14, os.path.join(repo14, "ARCHITECTURE.md"))
    expect(
        "exclude: a checkout below a 'build' directory hides nothing",
        (e, any("app must not include core" in v for v in out)),
        (1, True),
    )
    e, out = run(
        {
            "src/core/a.h": "#pragma once\n",
            "src/rebuild_queue/q.h": '#include "core/a.h"\n',
            "src/generated/g.h": '#include "core/a.h"\n',
            "src/core/legacy/old.h": '#include "rebuild_queue/q.h"\n',
        },
        contract(
            "scan: src\nexclude: build, gen*, src/core/legacy\n"
            "module core\nmodule rebuild_queue\nmodule generated\n"
        ),
    )
    expect(
        "exclude: names, globs and paths prune exactly - never substrings",
        (e, [v.strip() for v in out if "must not include" in v]),
        (1, ["rebuild_queue must not include core (x1; edge not declared)"]),
    )

    # 14b a scan root that does not exist is a contract error, never a skip
    e, out = run(
        {"src/core/a.h": "#pragma once\n", "include/app/b.h": '#include "core/a.h"\n'},
        contract("scan: src, incldue\nmodule core\nmodule app\n"),
    )
    expect(
        "a scan root that does not exist is a contract error",
        (e, any("do not exist" in v for v in out)),
        (2, True),
    )

    # 14c rooted includes: every root holding the first segment, declaration
    # order - the same verdict under every hash seed
    with tempfile.TemporaryDirectory(prefix="check_arch_seed_") as td:
        write(
            td,
            {
                "include/core/api.h": "#pragma once\n",
                "src/core/impl.h": "#pragma once\n",
                "src/app/main.cpp": "#include <core/impl.h>\n",
                "ARCHITECTURE.md": contract(
                    "scan: include, src\nmodule core\nmodule app\n"
                ),
            },
        )
        verdicts = set()
        for seed in range(8):
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--repo", td],
                env={**os.environ, "PYTHONHASHSEED": str(seed)},
                capture_output=True,
                text=True,
                check=False,
            )
            verdicts.add((proc.returncode, "app must not include core" in proc.stdout))
    expect("rooted includes resolve alike under 8 hash seeds", verdicts, {(1, True)})

    # 14d a quoted include whose rooted candidate is missing falls through
    e, out = run(
        {
            "src/detail/x.h": "#pragma once\n",
            "src/app/main.cpp": '#include "detail/util.h"\n',
            "src/app/detail/util.h": "#pragma once\n",
        },
        contract("scan: src\nmodule detail\nmodule app\nmodule app/detail\n"),
    )
    expect(
        "quoted include falls through to the quote chain",
        (e, any("app must not include app/detail" in v for v in out)),
        (1, True),
    )

    # 14e scan-deps over a synthetic scanner: one make rule, the TU first
    def scan_deps_run(files, body, deps, form="command", database=None, upper=False):
        seen = {}
        with tempfile.TemporaryDirectory(prefix="check_arch_sd_") as td:
            tu = os.path.join(td, *deps[0].split("/"))
            if database is None:
                if form == "arguments":
                    item = {
                        "directory": td,
                        "arguments": ["c++", "-fmodules-ts", "-c", tu],
                        "file": tu,
                    }
                else:
                    item = {
                        "directory": td,
                        "command": "c++ -fmodules-ts -c main.cpp",
                        "file": tu,
                    }
                database = json.dumps([item])
            paths = [os.path.join(td, *d.split("/")) for d in deps]
            mk = [(p.upper() if upper else p).replace(" ", "\\ ") for p in paths]
            stdout = "main.o: \\\n  " + " \\\n  ".join(mk) + "\n"

            def scanner(argv, **_kwargs):
                with open(argv[1].split("=", 1)[1], encoding="utf-8") as fh:
                    seen["db"] = json.load(fh)
                return subprocess.CompletedProcess(argv, 0, stdout, "")

            sd_knobs = (
                "graph: scan-deps\ndb: compile_commands.json\n"
                "clang_scan_deps: synthetic-scanner\n"
            )
            write(
                td,
                {
                    **files,
                    "compile_commands.json": database,
                    "ARCHITECTURE.md": contract(body + sd_knobs),
                },
            )
            with patch("subprocess.run", side_effect=scanner):
                e, out = run_check(td, os.path.join(td, "ARCHITECTURE.md"))
        return e, out, seen.get("db") or [{}]

    files_sd = {
        "src/app/main.cpp": '#include "shared/first.h"\n',
        "src/shared/first.h": '#pragma once\n#include "model/last.h"\n',
        "src/model/last.h": "#pragma once\n",
    }
    body_sd = (
        "scan: src\nmodule app\nmodule shared\nmodule model\n"
        "allow app -> shared\nallow app -> model\nallow shared -> model\n"
    )
    deps_sd = ["src/app/main.cpp", "src/shared/first.h", "src/model/last.h"]
    e, out, db = scan_deps_run(files_sd, body_sd, deps_sd)
    expect(
        "scan-deps: root-stripped module names classify like the regex mode",
        (
            e,
            [v for v in out if "unclassified" in v],
            "-fmodules-ts" in db[0]["command"],
        ),
        (0, [], False),
    )
    e, out, db = scan_deps_run(files_sd, body_sd, deps_sd, form="arguments")
    expect(
        "scan-deps: arguments-form entries are scrubbed in their own form",
        (e, "command" in db[0], "-fmodules-ts" in db[0].get("arguments", [])),
        (0, False, False),
    )
    e, out, _ = scan_deps_run(
        {**files_sd, "gen/config.h": "#pragma once\n"},
        body_sd + "forbidden: gen\n",
        [*deps_sd, "gen/config.h"],
    )
    expect(
        "scan-deps: an in-repo compiler-observed path matches forbidden",
        (
            e,
            any(
                "src/app/main.cpp: must not include gen/config.h (forbidden" in v
                for v in out
            ),
        ),
        (1, True),
    )
    if os.name == "nt":
        e, out, _ = scan_deps_run(files_sd, body_sd, deps_sd, upper=True)
        expect("scan-deps: the scanner's path case does not blind the graph", e, 0)
    files_obs = dict(files_sd)
    files_obs["src/shared/first.h"] = '#pragma once\n#include "last.h"\n'
    body_obs = (
        "scan: src\nmodule app\nmodule shared\nmodule model\nmodule extra\n"
        "allow app -> shared\nallow app -> model\nallow app -> extra\n"
    )
    e, out, _ = scan_deps_run(files_obs, body_obs, deps_sd)
    expect(
        "scan-deps: a bare include binds through the compiler-observed files",
        (e, any("shared must not include model" in v for v in out)),
        (1, True),
    )
    e, out, _ = scan_deps_run(
        {**files_obs, "src/extra/last.h": "#pragma once\n"},
        body_obs,
        [*deps_sd, "src/extra/last.h"],
    )
    expect(
        "scan-deps: an ambiguous observed include refuses the check",
        (e, any("ambiguous project include" in v for v in out)),
        (2, True),
    )
    for name, database, needle in (
        ("an unreadable", "{not json", "compile database unreadable"),
        ("a malformed", '{"file": 1}', "compile database malformed"),
    ):
        e, out, _ = scan_deps_run(files_sd, body_sd, deps_sd, database=database)
        expect(
            f"scan-deps: {name} compile database is refused, no traceback",
            (e, any(needle in v for v in out)),
            (2, True),
        )

    # 14f the make-deps parser: continuations, escapes, shared lines, rules
    make_cases = [
        ("/b/a.o: /s/a.cpp /s/a.h \\\n  /s/b.h\n", {"/s/a.cpp": {"/s/a.h", "/s/b.h"}}),
        (
            "main.obj: C:\\My\\ Repo\\src\\main.cpp \\\n  C:\\My\\ Repo\\src\\a.h\n",
            {"C:/My Repo/src/main.cpp": {"C:/My Repo/src/a.h"}},
        ),
        (
            "x.o: \\\n /s/x.cpp \\\n /s/x.h\ny.o: /s/y.cpp /s/y\\#1.h /s/$$z.h\n",
            {"/s/x.cpp": {"/s/x.h"}, "/s/y.cpp": {"/s/y#1.h", "/s/$z.h"}},
        ),
        (
            (
                "CMakeFiles\\g.dir\\v.cpp.obj: \\\n  C:\\R\\src\\v.cpp \\\n"
                "  C:\\Program\\ Files\\ (x86)\\Windows\\ Kits\\10\\stddef.h\n"
            ),
            {"C:/R/src/v.cpp": {"C:/Program Files (x86)/Windows Kits/10/stddef.h"}},
        ),
        # a path component that starts with a blank: 2k+1 backslashes -> k
        ("a.o: /s/a.cpp /s/dir\\\\\\ x.h\n", {"/s/a.cpp": {"/s/dir/ x.h"}}),
    ]
    expect(
        "make-deps parser: continuations, escapes, shared lines, rules",
        [parse_make_deps(text) for text, _ in make_cases],
        [want for _, want in make_cases],
    )

    # 14g truthful output: the clean line names only what was checked, and
    # the header-size report tier works with the smells off
    e, out = run(
        {
            "src/known/a.h": '#include "known/b.h"\n',
            "src/known/b.h": '#include "known/a.h"\n',
        },
        contract(b5 + "cycles: false\n"),
    )
    expect(
        "cycles: false - the clean line claims no cycle check",
        (e, any("no cycles" in v for v in out)),
        (0, False),
    )
    e, out = run(
        files12k, contract(b12k + "smells: off\nheader_size: report\n"), report=True
    )
    expect(
        "header_size report: listed with the smells off",
        (e, any("HEADER SIZE: src/big/1.h" in v for v in out)),
        (0, True),
    )

    # 14h derive never writes an empty draft; its draft passes its own check
    with tempfile.TemporaryDirectory(prefix="check_arch_nodraft_") as td:
        os.makedirs(os.path.join(td, "empty"))
        draft = os.path.join(td, "draft.md")
        with open(draft, "w", encoding="utf-8") as fh:
            fh.write("KEEP\n")
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            empty = main(["--repo", td, "--derive", "--scan", "empty", "--out", draft])
            absent = main(
                ["--repo", td, "--derive", "--scan", "absent", "--out", draft]
            )
        with open(draft, encoding="utf-8") as fh:
            kept = fh.read()
    expect(
        "derive: nothing to derive never overwrites a draft",
        (empty, absent, kept),
        (77, 2, "KEEP\n"),
    )
    with tempfile.TemporaryDirectory(prefix="check_arch_roundtrip_") as td:
        write(
            td,
            {
                "src/core/a.h": "#pragma once\n",
                "src/core/io/b.h": '#include "core/a.h"\n',
                "src/app/main.cpp": (
                    '#include "core/io/b.h"\n#include "third_party/lib/x.h"\n'
                    "#include <core/gone.h>\n"
                ),
                "src/third_party/lib/x.h": "#pragma once\n",
                "code/lib/util.h": "#pragma once\n",
                "code/lib/sub/s.h": '#include "../util.h"\n',
            },
        )
        lines, _warns = derive(
            td, ["src", "code/lib"], [".h", ".cpp"], DEFAULT_EXCLUDES
        )
        write(td, {"ARCHITECTURE.md": "\n".join(lines) + "\n"})
        e, out = run_check(td, os.path.join(td, "ARCHITECTURE.md"))
    knob = {
        ln.split(":", 1)[0]: [x.strip() for x in ln.split(":", 1)[1].split(",")]
        for ln in lines
        if ln.startswith(("exclude:", "external:"))
    }
    expect(
        "derive -> check: the unedited draft passes its own check",
        (e, [v for v in out if "must" in v or "unclassified" in v]),
        (0, []),
    )
    expect(
        "derive: vendored dirs excluded, project dirs never external",
        ("third_party" in knob.get("exclude", []), "core" in knob.get("external", [])),
        (True, False),
    )

    # 14i contract typos are loud (exit 2), never silent no-ops; each case
    # names its own message, so no other refusal can stand in for it
    for name, body, needle in (
        (
            "a duplicate knob",
            "scan: src\nscan: include\nmodule a\n",
            "duplicate knob 'scan'",
        ),
        (
            "an extension without its dot",
            "scan: src\nextensions: hpp\nmodule a\n",
            "extensions: each entry starts with '.'",
        ),
        (
            "a namespace_root that is no C++ namespace",
            "scan: src\nmodule a\nnamespace_root: .hpp\nnamespace_module: .hpp\n",
            "namespace_root: a C++ namespace expected",
        ),
        (
            "a namespace_module matching no module",
            "scan: src\nmodule a\nnamespace_root: proj\nnamespace_module: proj\n",
            "is neither a declared module nor the path prefix of one",
        ),
        (
            "namespace_require without the gate",
            "scan: src\nmodule a\nnamespace_require: .hpp\n",
            "namespace_require needs namespace_root",
        ),
    ):
        e, out = run(
            {"src/a/x.hpp": "int x;\n", "include/a/y.hpp": "int y;\n"}, contract(body)
        )
        expect(
            f"contract error: {name}",
            (e, any("contract error" in v and needle in v for v in out)),
            (2, True),
        )

    # 14j reversed diagram markers: refused by the check and by --diagram,
    # the document left untouched
    reversed_doc = (
        "# A\n\n"
        + DIAGRAM_END
        + "\n\n"
        + DIAGRAM_BEGIN
        + "\n\n```arch\n"
        + b5
        + "```\n"
    )
    e, out = run({"src/known/k.h": "#pragma once\n"}, reversed_doc)
    expect(
        "reversed diagram markers are a contract error",
        (e, any("diagram markers malformed" in v for v in out)),
        (2, True),
    )
    with tempfile.TemporaryDirectory(prefix="check_arch_markers_") as td:
        doc = os.path.join(td, "ARCHITECTURE.md")
        with open(doc, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(reversed_doc)
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            code = main(["--repo", td, "--doc", "ARCHITECTURE.md", "--diagram"])
        with open(doc, encoding="utf-8", newline="") as fh:
            after = fh.read()
    expect(
        "--diagram refuses reversed markers, the file untouched",
        (code, after == reversed_doc),
        (2, True),
    )

    # 14k --json stays JSON on a contract error
    with tempfile.TemporaryDirectory(prefix="check_arch_json_") as td:
        write(
            td,
            {
                "src/a/x.h": "#pragma once\n",
                "ARCHITECTURE.md": contract("scan: src\nmodule a\nallow a -> typo\n"),
            },
        )
        sink = io.StringIO()
        with contextlib.redirect_stdout(sink):
            code = main(["--repo", td, "--json"])
    try:
        parsed = json.loads(sink.getvalue())
    except ValueError:
        parsed = {}
    expect(
        "--json stays JSON on a contract error",
        (code, parsed.get("code"), parsed.get("violations", "absent")),
        (2, 2, None),
    )

    print(f"check_arch {CHECK_ARCH_VERSION} selftest:", "PASS" if ok else "FAIL")
    return 0 if ok else 2


def detect_roots(repo):
    """Derive's default scan roots: every top-level directory that contains
    sources - except the vendor/non-source names (VENDOR_DIRS): vendored
    trees are never architecture."""
    roots = []
    for name in sorted(os.listdir(repo)):
        p = os.path.join(repo, name)
        if (
            not os.path.isdir(p)
            or name.startswith(".")
            or name in VENDOR_DIRS
            or any(x in name for x in DEFAULT_EXCLUDES)
        ):
            continue
        for _dirpath, dn, fn in os.walk(p):
            dn[:] = [
                d
                for d in dn
                if d not in VENDOR_DIRS and not any(x in d for x in DEFAULT_EXCLUDES)
            ]
            if any(os.path.splitext(f)[1] in DEFAULT_EXTS for f in fn):
                roots.append(name.replace("\\", "/"))
                break
    return roots


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        prog="check_arch.py",
        description="the generic architecture gate (analyze, define, check; rules in "
        "the arch block incl. the layer stack, the diagram generated + "
        "drift-gated)",
    )
    ap.add_argument(
        "--version", action="version", version=f"check_arch {CHECK_ARCH_VERSION}"
    )
    ap.add_argument(
        "--repo",
        help="the repository root to check/derive "
        "(default: the directory above this script)",
    )
    ap.add_argument(
        "--doc",
        help="contract document; a relative path resolves against --repo "
        "(default: docs/ARCHITECTURE.md, ARCHITECTURE.md, tools/ARCHITECTURE.md)",
    )
    ap.add_argument(
        "--report",
        action="store_true",
        help="also print the stack and the measured module graph",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="machine-readable output: the check summary as JSON",
    )
    ap.add_argument(
        "--diagram",
        action="store_true",
        help="regenerate the marked diagram section from the rules",
    )
    ap.add_argument(
        "--derive",
        action="store_true",
        help="ANALYZE: draft an ARCHITECTURE.md from the measured graph",
    )
    ap.add_argument(
        "--scan",
        help="derive only: comma-separated scan roots "
        "(default: every top-level source dir)",
    )
    ap.add_argument(
        "--out", help="derive only: write the draft to FILE (default: stdout)"
    )
    ap.add_argument(
        "--selftest",
        action="store_true",
        help="offline PASS proof on synthetic fixtures",
    )
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    # abspath: the scan-deps graph speaks absolute paths (compile databases
    # list absolute TUs) - a relative repo root would silently blind it (A1)
    repo = os.path.abspath(args.repo) if args.repo else os.path.dirname(here)

    if args.derive:
        if args.scan:
            roots = [_norm_root(r.strip()) for r in args.scan.split(",") if r.strip()]
        else:
            roots = detect_roots(repo)
        if not roots:
            print("derive: no source roots found (pass --scan DIRS)")
            return EXIT_SKIP
        missing = [r for r in roots if not os.path.isdir(os.path.join(repo, r))]
        if missing:
            print(f"derive: scan root(s) {missing} do not exist under {repo}")
            return EXIT_USAGE
        lines, warns = derive(repo, roots, DEFAULT_EXTS, DEFAULT_EXCLUDES)
        for w in warns:
            print("warning: " + w, file=sys.stderr)
        if not lines:
            return EXIT_SKIP  # nothing derived: an existing draft stays untouched
        text = "\n".join(lines) + "\n"
        if args.out:
            with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
            print(f"derive: draft written -> {args.out}")
        else:
            sys.stdout.write(text)
        return EXIT_OK

    if args.diagram:
        try:
            doc_path, text = load_contract_text(repo, args.doc)
            written = write_diagram(doc_path, Contract(text, doc_path), text)
        except ContractError as e:
            print(f"contract error: {e}")
            return EXIT_USAGE
        if written:
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
