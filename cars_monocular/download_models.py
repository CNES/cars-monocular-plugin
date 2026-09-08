#!/usr/bin/env python
# coding: utf8
#
# Copyright (c) 2026 Centre National d'Etudes Spatiales (CNES).
#
# This file is part of CARS Monocular
# (see https://github.com/CNES/cars-monocular).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
This file contains the model downloading script for CARS Monocular
"""

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, LocalEntryNotFoundError

logger = logging.getLogger(__name__)
MODEL_FILENAME = "model.pt"
DEFAULT_MODEL_KEY = "vitl-normal"

MODEL_MAP = {
    "vitl-normal": "Ruicheng/moge-2-vitl-normal",
    "vitb-normal": "Ruicheng/moge-2-vitb-normal",
    "vits-normal": "Ruicheng/moge-2-vits-normal",
}


def get_models_dir() -> Path:
    """Return the plugin models directory and ensure it exists."""
    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / "applications" / "depth_map_generation" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_output_file(model_key: str) -> Path:
    """Return the destination filename for a model key."""
    return get_models_dir() / f"moge-2-{model_key}.pt"


def download_with_wget(repo_id: str, output_file: Path) -> bool:
    """
    Download model.pt from Hugging Face using wget as a fallback.
    Returns True when successful, False otherwise.
    """
    if shutil.which("wget") is None:
        logger.error(
            "wget is not available in PATH. Fallback download skipped."
        )
        return False

    url = f"https://huggingface.co/{repo_id}/resolve/main/{MODEL_FILENAME}"
    logger.info(f"Falling back to wget download from {url}")

    try:
        subprocess.run(
            ["wget", url, "-O", str(output_file)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        logger.info("Model successfully downloaded using wget")
        return True
    except subprocess.CalledProcessError as exception:
        stderr = (exception.stderr or "").strip()
        logger.error(f"wget download failed: {stderr or str(exception)}")
        return False
    except Exception as exception:
        logger.error(f"Unexpected error during wget fallback: {exception}")
        return False


def download_model(model_key: str) -> Path:
    """
    Download a MoGe2 model and store it in the plugin model directory.
    Falls back to wget when huggingface_hub download fails.
    """
    repo_id = MODEL_MAP[model_key]
    output_file = get_output_file(model_key)

    if output_file.exists():
        logger.info(f"Model already present: {output_file}")
        return output_file

    logger.info(f"Selected model: {model_key} (repo: {repo_id})")
    logger.info("Downloading model. This may take a while...")

    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id, filename=MODEL_FILENAME
        )
        shutil.copy(downloaded_path, output_file)
        logger.info("Model successfully downloaded")
        return output_file
    except LocalEntryNotFoundError:
        logger.error(
            f"Model file {MODEL_FILENAME} not found in repository {repo_id}."
        )
    except HfHubHTTPError as exception:
        logger.error(f"Network or HuggingFace Hub error: {exception}")
        logger.error(
            "Check your internet connection or repository accessibility."
        )
    except Exception as exception:
        logger.error(f"Unexpected error during model download: {exception}")

    if download_with_wget(repo_id, output_file):
        return output_file

    raise RuntimeError(f"Failed to download model {model_key} from {repo_id}")


def ensure_default_model_available() -> None:
    """
    Ensure the default model is available in the plugin model directory.
    This function never raises to avoid breaking plugin import.
    """
    output_file = get_output_file(DEFAULT_MODEL_KEY)
    if output_file.exists():
        return

    try:
        download_model(DEFAULT_MODEL_KEY)
    except Exception as exception:
        logger.warning(
            f"Automatic download of default MoGe2 model failed: {exception}"
        )


def main():
    """
    Main function of the model downloading script
    """
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="Download a MoGe2 model into the plugin model directory."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="vitl-normal",
        choices=["vitl-normal", "vitb-normal", "vits-normal"],
        help="Which MoGe2 model variant to download.",
    )

    args = parser.parse_args()

    model_key = args.model

    try:
        download_model(model_key)
    except Exception as exception:
        logger.error(f"Model download failed: {exception}")
        return 1

    return 0


if __name__ == "__main__":
    main()
