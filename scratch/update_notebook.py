import json
from pathlib import Path

notebook_path = Path("e:/VSCODE_WORKSPACE/NewDatathon/notebooks/compare_submissions.ipynb")

if notebook_path.exists():
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    changed = False
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            new_source = []
            for line in source:
                if "BEST_SUB_PATH =" in line and "best_750k.csv" in line:
                    new_line = line.replace("best_750k.csv", "best_624k.csv")
                    new_source.append(new_line)
                    changed = True
                    print(f"Updated line: {line.strip()} -> {new_line.strip()}")
                else:
                    new_source.append(line)
            cell['source'] = new_source
            
    if changed:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Notebook updated successfully.")
    else:
        print("Target line not found or already updated.")
else:
    print(f"Error: {notebook_path} not found.")
