#!/usr/bin/env python3
"""
HuggingFace Model Downloader for ONNX Models

This script downloads ONNX models from HuggingFace Hub with support for
different quantization variants (FULL/FP32, FP16, INT8, Q4).

Usage:
    python download_model.py --list
    python download_model.py --model <model_name> [--variants <variants>]
    python download_model.py --model <hf_repo> [--variants <variants>]

Examples:
    python download_model.py --list
    python download_model.py --model flan-t5-small
    python download_model.py --model google/flan-t5-small --variants int8,q4
    python download_model.py --model Xenova/flan-t5-base --variants full,int8
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Set
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Try to import huggingface_hub
try:
    from huggingface_hub import hf_hub_download, list_repo_files, HfApi
    from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
except ImportError:
    logger.error("huggingface_hub not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import hf_hub_download, list_repo_files, HfApi
    from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError


# Model registry with known ONNX models
MODEL_REGISTRY = {
    "flan-t5-small": {
        "repo": "Xenova/flan-t5-small",
        "description": "FLAN-T5 Small (60M params) - encoder-decoder for summarization/QA",
        "architecture": "t5",
        "size_mb": 300
    },
    "flan-t5-base": {
        "repo": "Xenova/flan-t5-base",
        "description": "FLAN-T5 Base (220M params) - encoder-decoder for summarization/QA",
        "architecture": "t5",
        "size_mb": 900
    },
    "flan-t5-large": {
        "repo": "dmmagdal/flan-t5-large-onnx",
        "description": "FLAN-T5 Large (770M params) - encoder-decoder for summarization/QA",
        "architecture": "t5",
        "size_mb": 3000
    },
    "distilbart": {
        "repo": "Xenova/distilbart-cnn-12-6",
        "description": "DistilBART (306M params) - encoder-decoder for summarization",
        "architecture": "bart",
        "size_mb": 1200
    },
    "qwen3": {
        "repo": "onnx-community/Qwen3-1.7B-ONNX",
        "description": "Qwen 3 1.7B Instruct - decoder-only chat model",
        "architecture": "qwen",
        "size_mb": 6000
    },
    "llama3": {
        "repo": "onnx-community/Llama-3.2-1B-Instruct-ONNX",
        "description": "Llama 3.2 1B Instruct - decoder-only chat model",
        "architecture": "llama",
        "size_mb": 5000
    },
    "phi3": {
        "repo": "microsoft/Phi-3-mini-128k-instruct-onnx",
        "description": "Phi-3 Mini 128k Instruct - decoder-only chat model",
        "architecture": "phi",
        "size_mb": 7000
    },
}

# Variant mapping for file patterns
VARIANT_PATTERNS = {
    "full": ["encoder_model.onnx", "decoder_model.onnx", "decoder_with_past_model.onnx",
             "model.onnx", "decoder_model_merged.onnx"],
    "fp16": ["_fp16.onnx", "-fp16.onnx"],
    "int8": ["_int8.onnx", "-int8.onnx", "_quantized.onnx"],
    "q4": ["_int4.onnx", "-int4.onnx", "_q4.onnx", "-q4.onnx"],
}

# Essential config files to always download
ESSENTIAL_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "vocab.txt",
    "sentencepiece.bpe.model",
    "tokenizer.model",
]


class ModelDownloader:
    """Downloads ONNX models from HuggingFace Hub with variant filtering"""

    def __init__(self, output_dir: str = None):
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).parent
        self.api = HfApi()

    def list_models(self):
        """List all available models in the registry"""
        print("\n" + "="*80)
        print("Available ONNX Models")
        print("="*80 + "\n")

        for name, info in MODEL_REGISTRY.items():
            print(f"  {name:<20} - {info['description']}")
            print(f"  {'':20}   Repo: {info['repo']}")
            print(f"  {'':20}   Size: ~{info['size_mb']}MB (FULL variant)")
            print()

        print("Usage:")
        print(f"  python {Path(__file__).name} --model <model_name> [--variants full,int8,q4]")
        print(f"  python {Path(__file__).name} --model <huggingface_repo>")
        print()

    def resolve_repo_id(self, model_name: str) -> str:
        """
        Resolve model name to HuggingFace repository ID.
        Supports both registry aliases and direct repo IDs.
        """
        # Check if it's a registry alias
        if model_name in MODEL_REGISTRY:
            return MODEL_REGISTRY[model_name]["repo"]

        # Check if it's already a valid repo ID (contains /)
        if "/" in model_name:
            return model_name

        # Check if it's a partial match in registry
        matches = [name for name in MODEL_REGISTRY.keys() if model_name.lower() in name.lower()]
        if matches:
            logger.info(f"Found partial match: {matches[0]}")
            return MODEL_REGISTRY[matches[0]]["repo"]

        raise ValueError(f"Unknown model: {model_name}. Use --list to see available models.")

    def get_repo_files(self, repo_id: str) -> List[str]:
        """Get list of all files in the repository"""
        try:
            files = list_repo_files(repo_id=repo_id)
            return files
        except RepositoryNotFoundError:
            raise ValueError(f"Repository not found: {repo_id}")
        except HfHubHTTPError as e:
            raise ValueError(f"Error accessing repository {repo_id}: {e}")

    def filter_files_by_variants(
        self,
        all_files: List[str],
        variants: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Filter files by requested variants and categorize them.
        Returns dict with 'onnx' and 'config' file lists.
        """
        if variants is None:
            variants = ["full", "fp16", "int8", "q4"]

        # Normalize variant names
        variants = [v.lower().strip() for v in variants]

        # Find ONNX model files
        onnx_files = []
        for file_path in all_files:
            if not file_path.endswith(".onnx") and not file_path.endswith(".onnx_data"):
                continue

            file_lower = file_path.lower()

            # Check each variant
            for variant in variants:
                if variant == "full":
                    # Full models don't have quantization suffixes
                    if not any(pattern in file_lower for patterns in VARIANT_PATTERNS.values()
                             for pattern in patterns if pattern != "encoder_model.onnx"
                             and pattern != "decoder_model.onnx"
                             and pattern != "decoder_with_past_model.onnx"
                             and pattern != "model.onnx"
                             and pattern != "decoder_model_merged.onnx"):
                        # It's a base model file without quantization suffix
                        if any(base in file_lower for base in ["encoder_model.onnx", "decoder_model.onnx",
                                                                "decoder_with_past_model.onnx", "model.onnx",
                                                                "decoder_model_merged.onnx"]):
                            onnx_files.append(file_path)
                            break
                else:
                    # Check if file matches variant patterns
                    patterns = VARIANT_PATTERNS.get(variant, [])
                    if any(pattern in file_lower for pattern in patterns):
                        onnx_files.append(file_path)
                        break

        # Find essential config files
        config_files = []
        for file_path in all_files:
            file_name = Path(file_path).name
            # Check if it's an essential file or in onnx subdirectory with essential name
            if file_name in ESSENTIAL_FILES:
                config_files.append(file_path)
            elif "onnx" in file_path.lower() and file_name in ESSENTIAL_FILES:
                config_files.append(file_path)

        # Remove duplicates while preserving order
        onnx_files = list(dict.fromkeys(onnx_files))
        config_files = list(dict.fromkeys(config_files))

        return {
            "onnx": onnx_files,
            "config": config_files
        }

    def download_files(
        self,
        repo_id: str,
        files: List[str],
        local_dir: Path,
        desc: str = "files"
    ) -> List[Path]:
        """Download files from repository to local directory"""
        downloaded = []

        for file_path in files:
            try:
                # Determine local path
                local_file = local_dir / file_path

                # Skip if already exists
                if local_file.exists():
                    logger.info(f"  ✓ Already exists: {file_path}")
                    downloaded.append(local_file)
                    continue

                # Create parent directories
                local_file.parent.mkdir(parents=True, exist_ok=True)

                # Download file
                logger.info(f"  ↓ Downloading: {file_path}")
                hf_hub_download(
                    repo_id=repo_id,
                    filename=file_path,
                    local_dir=local_dir
                )
                downloaded.append(local_file)

            except Exception as e:
                logger.warning(f"  ⚠ Failed to download {file_path}: {e}")

        return downloaded

    def download_model(
        self,
        model_name: str,
        variants: Optional[List[str]] = None
    ) -> Path:
        """
        Download a model with specified variants.

        Args:
            model_name: Model name (registry alias or HF repo ID)
            variants: List of variants to download (full, fp16, int8, q4)

        Returns:
            Path to downloaded model directory
        """
        # Resolve repository ID
        repo_id = self.resolve_repo_id(model_name)
        logger.info(f"Resolved model: {model_name} -> {repo_id}")

        # Determine local directory name
        if model_name in MODEL_REGISTRY:
            dir_name = f"{model_name}-HF"
        else:
            dir_name = repo_id.replace("/", "-") + "-HF"

        local_dir = self.output_dir / dir_name
        logger.info(f"Download directory: {local_dir}")

        # Get all repository files
        logger.info("Fetching repository file list...")
        all_files = self.get_repo_files(repo_id)
        logger.info(f"Found {len(all_files)} files in repository")

        # Filter files by variants
        filtered = self.filter_files_by_variants(all_files, variants)

        onnx_files = filtered["onnx"]
        config_files = filtered["config"]

        if not onnx_files:
            variant_str = ", ".join(variants) if variants else "all"
            logger.warning(f"No ONNX files found for variants: {variant_str}")
            logger.info("Available ONNX files in repository:")
            for f in all_files:
                if f.endswith(".onnx"):
                    logger.info(f"  - {f}")
            return local_dir

        # Print summary
        variant_str = ", ".join(variants) if variants else "all"
        print("\n" + "="*80)
        print(f"Downloading: {repo_id}")
        print(f"Variants: {variant_str}")
        print(f"Destination: {local_dir}")
        print("="*80 + "\n")

        print(f"ONNX Models ({len(onnx_files)} files):")
        for f in onnx_files:
            print(f"  - {f}")
        print()

        print(f"Config Files ({len(config_files)} files):")
        for f in config_files:
            print(f"  - {f}")
        print()

        # Download files
        logger.info("Downloading ONNX model files...")
        downloaded_onnx = self.download_files(repo_id, onnx_files, local_dir, "ONNX models")

        logger.info("Downloading config files...")
        downloaded_config = self.download_files(repo_id, config_files, local_dir, "config files")

        # Print summary
        total_downloaded = len(downloaded_onnx) + len(downloaded_config)
        print("\n" + "="*80)
        print(f"✓ Download Complete!")
        print(f"  Total files: {total_downloaded}")
        print(f"  Location: {local_dir}")
        print("="*80 + "\n")

        # Print usage example
        self._print_usage_example(dir_name, variants)

        return local_dir

    def _print_usage_example(self, model_dir: str, variants: Optional[List[str]]):
        """Print Java usage example"""
        variant = "INT8" if not variants or "int8" in variants else \
                  "Q4" if "q4" in variants else \
                  "FP16" if "fp16" in variants else "FULL"

        print("Java Usage Example:")
        print("-" * 80)
        print(f"""
ModelConfig config = ModelConfig.builder()
    .modelPath("models/{model_dir}")
    .variant(ModelVariant.{variant})
    .build();

try (OnnxInference inference = OnnxInference.create(config, GenerationConfig.DEFAULT)) {{
    InferenceResponse response = inference.generate("Summarize: Your text here...");
    System.out.println(response.getResponseText());
}}
""")
        print("-" * 80)


def parse_variants(variants_str: str) -> List[str]:
    """Parse comma-separated variant string"""
    if not variants_str:
        return None

    variants = [v.strip().lower() for v in variants_str.split(",")]
    valid_variants = ["full", "fp16", "int8", "q4"]

    for v in variants:
        if v not in valid_variants:
            raise ValueError(f"Invalid variant: {v}. Valid options: {', '.join(valid_variants)}")

    return variants


def main():
    parser = argparse.ArgumentParser(
        description="Download ONNX models from HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list
  %(prog)s --model flan-t5-small
  %(prog)s --model qwen3 --variants int8,q4
  %(prog)s --model Xenova/flan-t5-base --variants full,int8

Variants:
  full   - Full precision (FP32) - best quality, largest size
  fp16   - Half precision - 2x smaller, GPU optimized
  int8   - 8-bit quantization - 4x smaller, good quality
  q4     - 4-bit quantization - 8x smaller, acceptable quality
        """
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Model name (from registry) or HuggingFace repo ID (owner/model)"
    )

    parser.add_argument(
        "--variants",
        type=str,
        help="Comma-separated list of variants to download (full,fp16,int8,q4). Default: all"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for downloaded models (default: ./models)"
    )

    args = parser.parse_args()

    # Create downloader
    downloader = ModelDownloader(output_dir=args.output_dir)

    # Handle list command
    if args.list:
        downloader.list_models()
        return 0

    # Require model name
    if not args.model:
        parser.error("--model is required (or use --list to see available models)")

    # Parse variants
    try:
        variants = parse_variants(args.variants)
    except ValueError as e:
        parser.error(str(e))

    # Download model
    try:
        downloader.download_model(args.model, variants)
        return 0
    except Exception as e:
        logger.error(f"Download failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
