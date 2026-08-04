from __future__ import annotations

import ast
import builtins
import io
import os
import pickle
from collections.abc import Mapping
from typing import Any


DEFAULT_ALLOWED_IMPORT_ROOTS = frozenset(
    {
        "base64",
        "collections",
        "copy",
        "decimal",
        "datetime",
        "fractions",
        "functools",
        "gzip",
        "itertools",
        "lightgbm",
        "math",
        "numpy",
        "pandas",
        "pickle",
        "plotly",
        "random",
        "re",
        "scipy",
        "sklearn",
        "statistics",
        "statsmodels",
        "time",
        "torch",
        "torchvision",
        "xgboost",
    }
)

_BLOCKED_NAMES = frozenset(
    {
        "__import__",
        "breakpoint",
        "builtins",
        "compile",
        "ctypes",
        "delattr",
        "dir",
        "eval",
        "exec",
        "ftplib",
        "getattr",
        "glob",
        "globals",
        "help",
        "http",
        "importlib",
        "input",
        "locals",
        "memoryview",
        "multiprocessing",
        "open",
        "os",
        "pathlib",
        "requests",
        "setattr",
        "shutil",
        "socket",
        "subprocess",
        "sys",
        "tempfile",
        "threading",
        "urllib",
        "vars",
    }
)

_BLOCKED_ATTRIBUTES = frozenset(
    {
        "builtins",
        "ctypes",
        "dump",
        "download",
        "download_url_to_file",
        "environ",
        "ExcelFile",
        "execv",
        "execve",
        "fork",
        "fromfile",
        "HDFStore",
        "importlib",
        "io",
        "load",
        "load_model",
        "loads",
        "memmap",
        "popen",
        "read_bytes",
        "read_clipboard",
        "read_csv",
        "read_excel",
        "read_feather",
        "read_fwf",
        "read_gbq",
        "read_hdf",
        "read_html",
        "read_json",
        "read_orc",
        "read_parquet",
        "read_pickle",
        "read_sql",
        "read_sql_query",
        "read_sql_table",
        "read_sas",
        "read_spss",
        "read_stata",
        "read_table",
        "read_text",
        "read_xml",
        "rmdir",
        "save",
        "save_model",
        "savetxt",
        "savez",
        "savez_compressed",
        "sleep",
        "spawn",
        "subprocess",
        "system",
        "sys",
        "to_clipboard",
        "to_csv",
        "to_excel",
        "to_feather",
        "to_hdf",
        "to_html",
        "to_latex",
        "to_orc",
        "to_parquet",
        "to_pickle",
        "to_sql",
        "to_stata",
        "to_xml",
        "tofile",
        "touch",
        "unlink",
        "urlretrieve",
        "walk",
        "write_bytes",
        "write_html",
        "write_image",
        "write_json",
        "write_text",
    }
)

_SAFE_BUILTIN_NAMES = (
    "__build_class__",
    "ArithmeticError",
    "AssertionError",
    "Exception",
    "IndexError",
    "KeyError",
    "OverflowError",
    "RuntimeError",
    "StopIteration",
    "TypeError",
    "ValueError",
    "ZeroDivisionError",
    "abs",
    "all",
    "any",
    "bool",
    "bytearray",
    "bytes",
    "callable",
    "chr",
    "complex",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "hash",
    "hasattr",
    "hex",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "object",
    "oct",
    "ord",
    "pow",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "super",
    "tuple",
    "type",
    "zip",
)

_ALLOWED_PICKLE_ROOTS = frozenset(
    {
        "_codecs",
        "collections",
        "copyreg",
        "lightgbm",
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "statsmodels",
        "torch",
        "torchvision",
        "xgboost",
    }
)
_ALLOWED_PICKLE_BUILTINS = frozenset(
    {
        "bool",
        "bytearray",
        "bytes",
        "complex",
        "dict",
        "float",
        "frozenset",
        "int",
        "list",
        "object",
        "set",
        "slice",
        "str",
        "tuple",
    }
)


class UnsafeCodeError(ValueError):
    pass


def generated_code_execution_policy() -> dict[str, Any]:
    """Return the sandbox contract supplied to generated-code prompts.

    Keeping this contract next to the validator prevents prompts from drifting
    away from the rules enforced by :func:`safe_exec`.
    """
    return {
        "executor": "safe_exec",
        "allowed_import_roots": sorted(DEFAULT_ALLOWED_IMPORT_ROOTS),
        "allowed_builtin_names": sorted(_SAFE_BUILTIN_NAMES),
        "forbidden_names": sorted(_BLOCKED_NAMES),
        "forbidden_attributes": sorted(_BLOCKED_ATTRIBUTES),
        "rules": [
            "Do not use forbidden names or attributes; validation rejects them before execution.",
            "Do not use private or dunder attributes, except __call__ and __init__.",
            "Do not use dynamic reflection such as getattr; access a known public attribute directly.",
            "Do not use file, network, process, environment, or dynamic-code APIs.",
        ],
    }


class _SafetyVisitor(ast.NodeVisitor):
    def __init__(self, allowed_import_roots: frozenset[str]):
        self.allowed_import_roots = allowed_import_roots
        self.node_count = 0

    def generic_visit(self, node: ast.AST) -> None:
        self.node_count += 1
        if self.node_count > 12000:
            raise UnsafeCodeError("Generated code is too large.")
        super().generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_import(alias.name)
            self._check_identifier(alias.asname)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level:
            raise UnsafeCodeError("Relative imports are not allowed.")
        self._check_import(node.module or "")
        for alias in node.names:
            if alias.name == "*":
                raise UnsafeCodeError("Wildcard imports are not allowed.")
            self._check_identifier(alias.name)
            self._check_identifier(alias.asname)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in _BLOCKED_NAMES or "__" in node.id:
            raise UnsafeCodeError(f"Name '{node.id}' is not allowed.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            name = node.func.id
            if (
                name in _BLOCKED_ATTRIBUTES
                or name.startswith("download_")
                or name.startswith("fetch_")
            ):
                raise UnsafeCodeError(f"Call '{name}' is not allowed.")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        is_private = node.attr.startswith("_") or "__" in node.attr
        allowed_dunder = node.attr in {"__call__", "__init__"}
        if (
            (is_private and not allowed_dunder)
            or node.attr in _BLOCKED_ATTRIBUTES
            or node.attr.startswith("download_")
            or node.attr.startswith("fetch_")
        ):
            raise UnsafeCodeError(f"Attribute '{node.attr}' is not allowed.")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._check_identifier(node.name)
        if node.decorator_list or node.keywords:
            raise UnsafeCodeError(
                "Class decorators and metaclass options are not allowed."
            )
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_identifier(node.name)
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ):
            self._check_identifier(argument.arg)
        if node.args.vararg:
            self._check_identifier(node.args.vararg.arg)
        if node.args.kwarg:
            self._check_identifier(node.args.kwarg.arg)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        raise UnsafeCodeError("Async functions are not allowed.")

    def visit_While(self, node: ast.While) -> None:
        raise UnsafeCodeError("While loops are not allowed.")

    def visit_Global(self, node: ast.Global) -> None:
        raise UnsafeCodeError("Global declarations are not allowed.")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        raise UnsafeCodeError("Nonlocal declarations are not allowed.")

    def _check_import(self, module_name: str) -> None:
        root = module_name.split(".", 1)[0]
        if not root or root not in self.allowed_import_roots:
            raise UnsafeCodeError(f"Import '{module_name}' is not allowed.")

    @staticmethod
    def _check_identifier(name: str | None) -> None:
        if name and (
            name in _BLOCKED_NAMES
            or name in _BLOCKED_ATTRIBUTES
            or name.startswith("download_")
            or name.startswith("fetch_")
            or (
                "__" in name
                and name not in {"__call__", "__init__"}
            )
        ):
            raise UnsafeCodeError(f"Identifier '{name}' is not allowed.")


def validate_code(
    code: str,
    *,
    allowed_import_roots: frozenset[str] = DEFAULT_ALLOWED_IMPORT_ROOTS,
) -> ast.Module:
    source = str(code or "")
    if not source.strip():
        raise UnsafeCodeError("Generated code is empty.")
    if len(source) > 200_000:
        raise UnsafeCodeError("Generated code is too large.")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise UnsafeCodeError(f"Generated code has invalid syntax: {exc}") from exc
    _SafetyVisitor(allowed_import_roots).visit(tree)
    return tree


def _restricted_import(
    allowed_import_roots: frozenset[str],
):
    original_import = builtins.__import__

    def restricted_import(
        name: str,
        globals: Mapping[str, Any] | None = None,
        locals: Mapping[str, Any] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if level:
            raise ImportError("Relative imports are not allowed.")
        root = str(name or "").split(".", 1)[0]
        if root not in allowed_import_roots:
            raise ImportError(f"Import '{name}' is not allowed.")
        return original_import(name, globals, locals, fromlist, level)

    return restricted_import


def safe_builtins(
    *,
    allowed_import_roots: frozenset[str] = DEFAULT_ALLOWED_IMPORT_ROOTS,
) -> dict[str, Any]:
    values = {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES}
    values["__import__"] = _restricted_import(allowed_import_roots)
    return values


def safe_exec(
    code: str,
    namespace: dict[str, Any],
    *,
    allowed_import_roots: frozenset[str] = DEFAULT_ALLOWED_IMPORT_ROOTS,
) -> dict[str, Any]:
    tree = validate_code(code, allowed_import_roots=allowed_import_roots)
    namespace.setdefault("__name__", "__autostat_generated__")
    namespace["__builtins__"] = safe_builtins(
        allowed_import_roots=allowed_import_roots
    )
    exec(compile(tree, "<autostat-generated-code>", "exec"), namespace)
    return namespace


def safe_subprocess_env() -> dict[str, str]:
    allowed_names = (
        "CUDA_VISIBLE_DEVICES",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "MKL_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PATH",
        "PYTHONHOME",
        "PYTHONPATH",
        "SYSTEMROOT",
        "WINDIR",
    )
    env = {
        name: value
        for name in allowed_names
        if (value := os.environ.get(name))
    }
    temp_dir = (
        os.environ.get("AUTOSTAT_CODE_TMPDIR")
        or os.environ.get("TEMP")
        or os.environ.get("TMP")
        or "/tmp"
    )
    env["HOME"] = os.environ.get("AUTOSTAT_CODE_HOME") or temp_dir
    env["TMPDIR"] = temp_dir
    env["PYTHONUNBUFFERED"] = "1"
    return env


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        root = module.split(".", 1)[0]
        if module == "builtins" and name in _ALLOWED_PICKLE_BUILTINS:
            return super().find_class(module, name)
        if root in _ALLOWED_PICKLE_ROOTS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(
            f"Model artifact references disallowed global '{module}.{name}'."
        )


def restricted_pickle_loads(data: bytes) -> Any:
    return _RestrictedUnpickler(io.BytesIO(data)).load()
