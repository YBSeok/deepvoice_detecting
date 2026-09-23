#!/usr/bin/env python3
"""Upload v3.1 ad_lora to Hugging Face org.

  set HF_TOKEN=hf_xxx   # PowerShell: $env:HF_TOKEN='hf_xxx'
  python upload_ad_lora_hf.py
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import HfApi, login

REPO_ID = "ssafy19-dream/deepvoice-pjs-v3.1-ad-lora"
FOLDER = Path(__file__).resolve().parent / "hf_upload" / "deepvoice-pjs-v3.1-ad-lora"


def main():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        login(token=token, add_to_git_credential=False)
    api = HfApi()
    who = api.whoami()
    print("logged in as:", who.get("name") or who)
    orgs = [o.get("name") for o in (who.get("orgs") or [])]
    print("orgs:", orgs)
    if "ssafy19-dream" not in orgs and who.get("name") != "ssafy19-dream":
        print("WARNING: ssafy19-dream not listed in orgs — create may fail without write access")

    if not (FOLDER / "ad_lora.pt").is_file():
        raise SystemExit(f"missing {FOLDER / 'ad_lora.pt'}")

    api.create_repo(REPO_ID, repo_type="model", exist_ok=True, private=False)
    info = api.upload_folder(
        folder_path=str(FOLDER),
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Add v3.1 AntiDeepfake LoRA (ad_lora.pt)",
    )
    print("uploaded:", info)
    print(f"https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
