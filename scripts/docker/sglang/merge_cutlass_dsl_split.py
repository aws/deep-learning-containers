"""Reunite the CuTe DSL package, which pip splits across lib/ and lib64/ on AL2023.

nvidia-cutlass-dsl's payload arrives in two wheels that disagree on their wheel tag while
writing into one shared directory tree:

  nvidia-cutlass-dsl-libs-core  Root-Is-Purelib: true   -> .../lib/python3.X/site-packages
  nvidia-cutlass-dsl-libs-base  Root-Is-Purelib: false  -> .../lib64/python3.X/site-packages

Both target nvidia_cutlass_dsl/dsl_packages/cutlass/, so cutlass/__init__.py lands under lib
while cutlass/_mlir/ lands under lib64. nvidia-cutlass-dsl itself is only a .pth that does
`sys.path.insert(0, nvidia_cutlass_dsl.__path__[0] + '/dsl_packages')` -- a single path, so
whichever half wins is the only half importable, and `import cutlass._mlir` fails.

This is an upstream packaging bug, not a version problem: 4.6.0 and 4.7.0 are both affected.
It is also invisible on distros where purelib and platlib are the same directory (Debian,
Ubuntu, most manylinux images) and only breaks on the lib/lib64-split ones -- AL2023, RHEL,
Fedora -- which is why it ships broken and why it lands on us.

Symlinks rather than copies: the _mlir extension modules are large, and a symlink keeps pip's
RECORD accurate for the real file so a later uninstall or reinstall of either wheel still
does the right thing.

Idempotent, and asserts the merge achieved something -- a silent no-op here would just move
the failure to the import assertion in the Dockerfile with a more confusing message.
"""

import os
import sys
import sysconfig

REL = os.path.join("nvidia_cutlass_dsl", "dsl_packages")

purelib = sysconfig.get_paths()["purelib"]
platlib = sysconfig.get_paths()["platlib"]

if os.path.realpath(purelib) == os.path.realpath(platlib):
    print(f"cutlass DSL merge: purelib == platlib ({purelib}), nothing to merge")
    sys.exit(0)

linked = 0
# Link both directions: which half holds cutlass/__init__.py (and therefore which one the
# .pth resolves to) depends on install order, so do not assume lib64 is the orphan.
for src_root, dst_root in ((platlib, purelib), (purelib, platlib)):
    src = os.path.join(src_root, REL)
    if not os.path.isdir(src):
        continue
    for dirpath, _dirnames, filenames in os.walk(src):
        rel = os.path.relpath(dirpath, src)
        dst_dir = os.path.join(dst_root, REL, rel) if rel != "." else os.path.join(dst_root, REL)
        if not os.path.isdir(dst_dir):
            os.makedirs(dst_dir, exist_ok=True)
        for name in filenames:
            dst = os.path.join(dst_dir, name)
            if os.path.lexists(dst):
                continue
            os.symlink(os.path.join(dirpath, name), dst)
            linked += 1

print(f"cutlass DSL merge: linked {linked} file(s) across purelib/platlib")

# Verify by structure rather than by link count: a rerun after a successful merge legitimately
# links nothing, so a zero count is not itself a failure. What must hold is that both halves
# now sit under the same root, since the .pth only ever puts one of them on sys.path.
missing = [
    root
    for root in (purelib, platlib)
    if os.path.isdir(os.path.join(root, REL, "cutlass"))
    and not (
        os.path.exists(os.path.join(root, REL, "cutlass", "__init__.py"))
        and os.path.isdir(os.path.join(root, REL, "cutlass", "_mlir"))
    )
]
if missing:
    print(
        "ERROR: cutlass/__init__.py and cutlass/_mlir/ are still not co-located under "
        + ", ".join(missing)
        + " -- the wheel layout changed, or only one of nvidia-cutlass-dsl-libs-base / "
        "-libs-core is installed.",
        file=sys.stderr,
    )
    sys.exit(1)
