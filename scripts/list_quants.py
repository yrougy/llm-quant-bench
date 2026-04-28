#!/usr/bin/env python3
import sys
from huggingface_hub import list_repo_files

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <repo_id>")
    sys.exit(1)

repo = sys.argv[1]
for f in list_repo_files(repo):
    if f.endswith('.gguf'):
        print(f)
