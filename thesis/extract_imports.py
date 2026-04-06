import os
import ast
import json

PROJECT_ROOT = "."

imports = set()

def extract_from_py(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            tree = ast.parse(f.read())
        except:
            return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.add(n.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])

def extract_from_ipynb(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            nb = json.load(f)
        except:
            return

    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            try:
                tree = ast.parse(source)
            except:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for n in node.names:
                        imports.add(n.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split(".")[0])

for root, _, files in os.walk(PROJECT_ROOT):
    for file in files:
        path = os.path.join(root, file)
        if file.endswith(".py"):
            extract_from_py(path)
        elif file.endswith(".ipynb"):
            extract_from_ipynb(path)

print("\n".join(sorted(imports)))
