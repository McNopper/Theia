# Architecture

The checked-in architecture contract for Theia (the check-arch reflexion
gate). One leaf (`theia/scene`), the renderer consuming it, and the thin
`demo` host on top. **Theia never depends on Hyperion** — the sibling
renderers are independent; shared code goes down into Harmonia (a
`hyperion/` include is a hard violation).

<!-- check_arch: diagram BEGIN -->
### the layers

```mermaid
block-beta
    columns 3
    host["host"]:3
    renderer["renderer"]:2
    space
    scene["scene"]
    space:2
```

### layer host - the modules

```mermaid
block-beta
    columns 3
    demo["demo"]:3
```

### layer renderer - the modules

```mermaid
block-beta
    columns 2
    theia_renderer["theia/renderer"]:2
```

### layer scene - the modules

```mermaid
block-beta
    columns 1
    theia_scene["theia/scene"]
```
<!-- check_arch: diagram END -->

## The machine contract

```arch
scan: src
extensions: .hpp, .cpp
exclude: build, _deps, .git
unclassified: error
external: SDL3, harmonia, slang-math, vma, volk
forbidden: hyperion
quotes_relative: true
repo_fallback: false
cycles: true
namespace_root:
namespace_module:
namespace_require:
unresolved: info
within_layer: deny              # undeclared same-layer edges: violations
header_size: warn                # the standing bloat rule (promote to error when the queue is drained)
smells: report                 # refactoring-catalog proxies: report-only review signals

layer scene
layer renderer
layer host

module theia/scene in scene
module theia/renderer in renderer
module demo in host

allow theia/renderer -> theia/scene
allow demo -> theia/renderer
allow demo -> theia/scene
```
