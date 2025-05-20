from pathlib import Path
from sys import argv

import papermill as pm

PROJECT_ROOT = Path(__file__).parent
GENERATED_PATH = PROJECT_ROOT / "generated"


def generate_num_embedding_analysis(model_id):
    if not GENERATED_PATH.exists():
        GENERATED_PATH.mkdir(parents=True)

    notebook_path = GENERATED_PATH / f"{model_id.replace('/', '__')}.ipynb"

    if notebook_path.exists():
        notebook_path.unlink()

    pm.execute_notebook(
        str(PROJECT_ROOT / "00_papermill_template.ipynb"),
        str(notebook_path),
        parameters=dict(model_id=model_id),
    )


def main():
    if len(argv) < 2:
        print("Usage: python generate_notebook.py <model_id>")
        exit(1)
    model_id = argv[1]
    generate_num_embedding_analysis(model_id)
    print(f"Generated notebook for {model_id} at {GENERATED_PATH}")


if __name__ == "__main__":
    main()
