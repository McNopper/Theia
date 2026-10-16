#!/usr/bin/env python3
"""Coverage orchestrator for the THEIA_COVERAGE lane.

Expects a configure with -DTHEIA_COVERAGE=ON (a separate build dir, e.g.
build/coverage, keeps the main build untouched). Then this script builds the
instrumented tree, runs the tests with LLVM_PROFILE_FILE set, merges the
profiles and prints the llvm-cov report for the repo's own sources.

Exit codes: 0 report produced, 1 test or coverage failure, 77 skipped
(llvm-profdata/llvm-cov missing), 2 usage/config problems.

Usage: python tools/coverage.py [--build DIR] [--skip-build] [--show]
"""
import argparse
import io
import glob
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BUILD = os.path.join(ROOT, 'build', 'coverage')
PREFIX = 'THEIA_COVERAGE'

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def cmake_cmd():
    return 'cmake'


def llvm_tool(name, build_dir):
    # prefer the compiler's sibling directory (clang-cl ships the llvm tools)
    comp = None
    cache = os.path.join(build_dir, 'CMakeCache.txt')
    if os.path.isfile(cache):
        for line in io.open(cache, encoding='utf-8', errors='replace'):
            if line.startswith('CMAKE_CXX_COMPILER:') and ':FILEPATH=' in line:
                comp = line.split('FILEPATH=', 1)[1].strip()
                break
    if comp:
        cand = os.path.join(os.path.dirname(comp), name + '.exe')
        if os.path.isfile(cand):
            return cand
        cand = os.path.splitext(cand)[0]  # non-Windows layouts
        if os.path.isfile(cand):
            return cand
    return shutil.which(name)


def run(cmd, **kw):
    print('+', ' '.join(cmd))
    return subprocess.run(cmd, **kw).returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', default=DEFAULT_BUILD)
    ap.add_argument('--skip-build', action='store_true')
    ap.add_argument('--show', action='store_true', help='llvm-cov show instead of report')
    args = ap.parse_args()

    profdir = os.path.join(args.build, 'coverage-profiles')
    os.makedirs(profdir, exist_ok=True)

    if not args.skip_build:
        cmd = [cmake_cmd(), '-S', ROOT, '-B', args.build, '-D' + PREFIX + '=ON']
        # forward the main build's generator/toolchain settings so the coverage
        # configure resolves dependencies and compilers the same way (it stays
        # otherwise independent)
        main_cache = os.path.join(ROOT, 'build', 'CMakeCache.txt')
        if os.path.isfile(main_cache):
            forward = {}
            keys = ('CMAKE_GENERATOR:', 'CMAKE_GENERATOR_PLATFORM:',
                    'CMAKE_C_COMPILER:', 'CMAKE_CXX_COMPILER:',
                    'CMAKE_TOOLCHAIN_FILE:')
            for line in io.open(main_cache, encoding='utf-8', errors='replace'):
                for key in keys:
                    if line.startswith(key):
                        value = line.split('=', 1)[1].strip()
                        if value:
                            forward[key[:-1]] = value
                        break
            for name, value in forward.items():
                cmd += ['-D' + name + '=' + value]
        rc = run(cmd)
        if rc:
            return rc
        rc = run([cmake_cmd(), '--build', args.build])
        if rc:
            return rc

    profdata = llvm_tool('llvm-profdata', args.build)
    llvmcov = llvm_tool('llvm-cov', args.build)
    if not profdata or not llvmcov:
        print('coverage: llvm-profdata/llvm-cov not found (exit 77)')
        return 77

    env = dict(os.environ, LLVM_PROFILE_FILE=os.path.join(profdir, '%p.profraw'))
    rc = run(['ctest', '--test-dir', args.build, '--output-on-failure', '-LE', 'analysis'],
             env=env)
    if rc:
        return rc

    profraws = sorted(glob.glob(os.path.join(profdir, '*.profraw')))
    if not profraws:
        print('coverage: no .profraw files produced')
        return 1
    merged = os.path.join(profdir, 'merged.profdata')
    rc = run([profdata, 'merge', '-sparse'] + profraws + ['-o', merged])
    if rc:
        return rc

    srcdir = os.path.join(ROOT, 'src')
    exes = sorted(glob.glob(os.path.join(args.build, '**', '*.exe'), recursive=True))
    exes = [e for e in exes
            if os.sep + 'vcpkg_installed' + os.sep not in e
            and os.sep + 'CMakeFiles' + os.sep not in e and os.path.isfile(e)]
    failed = False
    for exe in exes:
        mode = 'show' if args.show else 'report'
        cmd = [llvmcov, mode, exe, '-instr-profile=' + merged, '-sources', srcdir]
        print('+', ' '.join(cmd))
        rc = subprocess.run(cmd).returncode
        failed = failed or rc != 0
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
