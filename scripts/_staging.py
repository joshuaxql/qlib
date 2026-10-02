"""Temporary build directories that inherit their dataset parent's access rules."""

from contextlib import contextmanager
from pathlib import Path
import shutil
from uuid import uuid4


@contextmanager
def staging_directory(root, *, prefix):
    """Create a unique child, retaining normal Windows ACL inheritance.

    TemporaryDirectory creates mode 0o700, whose Windows DACL excludes other
    readers. Renaming its descendants does not restore destination inheritance.
    These staging directories use mode 0o755 and remain inside the given root.
    """
    root = Path(root).expanduser().resolve()
    stage = root / f"{prefix}{uuid4().hex}"
    if stage.resolve().parent != root:
        raise ValueError("暂存目录超出指定根目录")
    stage.mkdir(mode=0o755, exist_ok=False)
    try:
        yield stage
    finally:
        if stage.resolve() != stage or stage.parent != root:
            raise ValueError("清理目录超出原暂存目录")
        shutil.rmtree(stage)
