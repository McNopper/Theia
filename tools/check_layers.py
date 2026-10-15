#!/usr/bin/env python3
"""Layer/architecture rules for theia (the code-quality-matrix "arch rules" cell).

Checks, over src/**:
  1. #include edges between modules respect the layer order - `theia/scene` is
     the leaf, `theia/renderer` consumes it, `demo` consumes both (the DAG is
     the design contract; cross-repo includes of harmonia/slang-math/aether
     are downward by design and only reported);
  2. no cyclic #include dependencies, detected as file-level strongly-connected
     components (catches same-layer header cycles a pairwise matrix misses).

Exit codes: 0 = clean, 1 = violations, 77 = skipped (surface not found).
Usage: python tools/check_layers.py [--report]
"""
import os
import re
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Leaf-first: a module may include only modules EARLIER in this list (or itself).
ORDER = ['theia', 'theia/scene', 'theia/renderer', 'demo']
ROOTS = {'theia/': 'src', 'demo/': 'src'}
EXTERNAL = ('harmonia/', 'slang-math/', 'aether/')
SCAN = ['src']

INC = re.compile(r'^\s*#\s*include\s+([<"])([\w./-]+)[>"]')


def module_of(rel):
    parts = rel.split(os.sep)[1:]  # strip the scan dir ('src')
    if parts[0] == 'demo':
        return 'demo'
    return '/'.join(parts[:2]) if len(parts) > 2 else parts[0]


def resolve(inc, from_file):
    for prefix, base in ROOTS.items():
        if inc.startswith(prefix):
            return os.path.normpath(os.path.join(ROOT, base, inc))
    return os.path.normpath(os.path.join(os.path.dirname(from_file), inc))


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


def main():
    report = '--report' in sys.argv
    files = {}
    for scan in SCAN:
        for dp, _, fs in os.walk(os.path.join(ROOT, scan)):
            if any(x in dp for x in ('build', '_deps', '.git')):
                continue
            for f in fs:
                if f.endswith(('.hpp', '.cpp', '.h', '.c')):
                    p = os.path.normpath(os.path.join(dp, f))
                    files[p] = os.path.relpath(p, ROOT)
    if not files:
        print(f'skip: no sources under {SCAN} in {ROOT}')
        return 77
    adj = {p: set() for p in files}
    mod_edges = {}
    ext = {}
    unresolved = []
    for p, rel in files.items():
        for line in open(p, encoding='utf-8', errors='replace'):
            m = INC.match(line)
            if not m:
                continue
            inc = m.group(2)
            if any(inc.startswith(x) for x in EXTERNAL):
                ext[inc.split('/')[0]] = ext.get(inc.split('/')[0], 0) + 1
                continue
            if m.group(1) == '<' and not any(inc.startswith(pfx) for pfx in ROOTS):
                continue  # angle-bracket outside the project roots: system header
            t = resolve(inc, p)
            if t in adj and t != p:
                adj[p].add(t)
                a, b = module_of(rel), module_of(files[t])
                if a != b:
                    mod_edges.setdefault((a, b), 0)
                    mod_edges[(a, b)] += 1
            elif t not in adj:
                unresolved.append(f'{rel}: "{inc}"')
    rank = {m: i for i, m in enumerate(ORDER)}
    violations = []
    for (a, b), n in sorted(mod_edges.items()):
        if a not in rank or b not in rank:
            violations.append(f'module {a} -> {b}: unclassified module')
        elif rank[a] < rank[b]:
            violations.append(f'{a} must not include {b} (upward edge, x{n})')
    cycles = [c for c in scc(list(files), adj) if len(c) > 1]
    for c in cycles:
        names = sorted(files[x] for x in c)
        violations.append('include cycle: ' + ' <-> '.join(names))
    if report:
        print('measured module edges (leaf-first order):')
        for m in ORDER:
            print(f'  {m}')
        for (a, b), n in sorted(mod_edges.items()):
            print(f'  {a} -> {b}  x{n}')
        if ext:
            print('external (downstream repos, allowed): ' +
                  ', '.join(f'{k} x{v}' for k, v in sorted(ext.items())))
        if unresolved:
            print('unresolved internal-looking includes (info only):')
            for u in unresolved[:10]:
                print(f'  {u}')
    if violations:
        print(f'{len(violations)} layer violation(s):')
        for v in violations:
            print('  ' + v)
        return 1
    print(f'layers clean ({len(files)} files, {len(mod_edges)} distinct module edges, '
          f'no cycles)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
