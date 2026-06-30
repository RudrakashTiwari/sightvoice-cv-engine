"""Run the full pipeline on a local image without starting the server.

    python tests/run_local.py path/to/image.jpg [language]
"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perception.pipeline import Pipeline

path = sys.argv[1]
language = sys.argv[2] if len(sys.argv) > 2 else "en"
with open(path, "rb") as f:
    result = Pipeline().run(f.read(), language=language)
print(json.dumps(result, indent=2, ensure_ascii=False))
