"""Compare recorded source hashes independently of the line-ending convention.

The experiment drivers record the SHA-256 of each source file's raw bytes on the
machine that ran them. Git stores these text files with LF line endings, yet a
working copy can hold CRLF instead (a Windows checkout, or an editor that saves
that way), so the raw bytes of identical code differ between systems. The
verification scripts therefore accept a recorded digest that matches the file
under either convention. Only line endings are ignored: any other difference in
the file still changes both digests and fails the check.
"""
import hashlib


def source_digests(path):
    """Return the SHA-256 digests of a text file with LF and with CRLF line endings."""
    unix = path.read_bytes().replace(b"\r\n", b"\n")
    return {hashlib.sha256(data).hexdigest() for data in (unix, unix.replace(b"\n", b"\r\n"))}


def matches_recorded_digest(path, recorded):
    """True when `recorded` is the digest of this file's content under either convention."""
    return recorded in source_digests(path)
