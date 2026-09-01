"""Verify Python's optional stdlib C extensions are present.

Python is built from source, and configure silently skips any extension whose
headers were missing at build time -- it reports no error, so a dropped module
would otherwise ship unnoticed.
"""

import pytest

OPTIONAL_MODULES = {
    "bz2": "bzip2-devel",
    "ctypes": "libffi-devel",
    "curses": "ncurses-devel",
    "hashlib": "openssl-devel",
    "lzma": "xz-devel",
    "readline": "readline-devel",
    "sqlite3": "sqlite-devel",
    "ssl": "openssl-devel",
    "zlib": "zlib-devel",
}


@pytest.mark.parametrize("module,package", OPTIONAL_MODULES.items())
def test_module_importable(module, package):
    """Import the module rather than checking importlib.find_spec.

    sqlite3 is a pure-Python package wrapping the _sqlite3 extension, so
    find_spec succeeds even when the extension is missing and the import fails.
    """
    try:
        __import__(module)
    except ImportError as exc:
        pytest.fail(f"import {module} failed ({exc}); Python needs {package} at build time")


def test_sqlite3_read_write():
    import sqlite3

    with sqlite3.connect(":memory:") as con:
        con.execute("create table t (a int, b text)")
        con.execute("insert into t values (?, ?)", (1, "x"))
        assert con.execute("select a, b from t").fetchall() == [(1, "x")]
