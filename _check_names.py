import ast
from pathlib import Path

IGNORE = {
    "print", "len", "float", "int", "str", "dict", "list", "set", "range", "max", "min",
    "sum", "getattr", "isinstance", "enumerate", "zip", "open", "Exception", "RuntimeError",
    "FileNotFoundError", "SystemExit", "Path", "torch", "np", "nn", "DataLoader",
    "TensorDataset", "argparse", "hashlib", "os", "random", "json", "csv", "tqdm",
    "VoiceFakeFusion", "fusion_features", "parse_rewrites", "rewrite_row", "row_has_voice",
    "build_split_rows", "is_domain_focus_row", "strip_aug_prefix", "FUSION_FEATURE_DIM",
}


def check(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    defs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    calls = {
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    missing = sorted(calls - defs - IGNORE)
    print(path)
    print("  unresolved:", missing)


check(r"D:\deepvoice_detecting\eval_ab_vf.py")
check(r"D:\deepvoice_detecting\train_fusion.py")
