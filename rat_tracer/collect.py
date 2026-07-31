"""``rat_tracer-collect``: package every marked frame into one archive file.

Run with no arguments. The researcher never has to know, type or navigate to
the storage directory -- the point of the command is that the path is the
application's business, not theirs. The archive lands somewhere a file manager
opens by default (Desktop, else Documents, else the home directory) so it can
be attached to a message without hunting through hidden or OS-internal
directories.

Archival is non-destructive: nothing on the researcher's machine is deleted or
moved, and running it twice leaves two archives, not one.
"""

import argparse
import socket
import sys
import zipfile
from datetime import datetime
from logging import getLogger
from pathlib import Path

from rat_tracer.bad_frames import configure_application_identity, storage_root
from rat_tracer.translations import resolve_translations

logger = getLogger(__name__)

#: Zip rather than tar.gz: every OS the researcher might use opens it natively
#: with no extra tooling, and they are the one who has to handle the file.
ARCHIVE_SUFFIX = ".zip"


def destination_directory() -> Path:
    """A directory the researcher can find in a file manager."""
    from PySide6.QtCore import QStandardPaths

    configure_application_identity()
    for location in (
        QStandardPaths.StandardLocation.DesktopLocation,
        QStandardPaths.StandardLocation.DocumentsLocation,
        QStandardPaths.StandardLocation.HomeLocation,
    ):
        candidate = QStandardPaths.writableLocation(location)
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    return Path.home()


def archive_name(now: datetime | None = None) -> str:
    """Name an archive so successive rounds stay distinct and attributable."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"rat_tracer_bad_frames_{socket.gethostname()}_{stamp}{ARCHIVE_SUFFIX}"


def collect(root: Path, destination: Path) -> Path:
    """Write everything under *root* into one archive inside *destination*.

    Returns the archive path. The archive is never written inside *root*, which
    would make each run include the previous one's output.
    """
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / archive_name()
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name.endswith(".tmp"):
                continue
            archive.write(path, path.relative_to(root))
    return archive_path


def has_marks(root: Path) -> bool:
    return root.is_dir() and any(
        path.is_file() and not path.name.endswith(".tmp") for path in root.rglob("*")
    )


def main() -> int:
    from PySide6.QtCore import QLocale

    strings = resolve_translations(QLocale.system().name())
    parser = argparse.ArgumentParser(description=strings["collect_description"])
    # No positional arguments by design; these two exist for the technician and
    # for tests, not for the researcher's one-command workflow.
    parser.add_argument("--source", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--destination", type=Path, default=None, help=argparse.SUPPRESS)
    # Read at call time, not via a bound import: the entry point passes
    # nothing, and tests drive the command by replacing sys.argv.
    args = parser.parse_args(sys.argv[1:])

    root = args.source if args.source is not None else storage_root()
    if not has_marks(root):
        print(strings["collect_nothing_to_archive"].format(path=root))
        return 1

    destination = args.destination if args.destination is not None else destination_directory()
    archive_path = collect(root, destination)
    print(strings["collect_done"].format(path=archive_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
