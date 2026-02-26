from __future__ import annotations

import multiprocessing
import os
import shutil

import numpy as np
from tqdm import tqdm

from BoCorpusQC.perplexity import PerplexityCalculator

# ---------------------------------------------------------------------------
# Module-level globals used by multiprocessing workers.
# They are populated by _init_worker via the Pool initializer.
# ---------------------------------------------------------------------------
_calculator: PerplexityCalculator | None = None


def _init_worker(tokenizer_type: str):
    """Pool initializer — each worker reconstructs its own PerplexityCalculator."""
    global _calculator
    print(f"Initializing worker (PID: {os.getpid()})...")
    if tokenizer_type == "sentencepiece":
        _calculator = PerplexityCalculator.from_sentencepiece()
    elif tokenizer_type == "syllable":
        _calculator = PerplexityCalculator.from_syllable()
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")


def _process_file(filepath: str):
    """Worker function — score a single file using the worker-local calculator."""
    global _calculator
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        if content.strip():
            ppl = _calculator.calculate_perplexity(content)
            return (filepath, ppl)
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

    return None


class DocumentFilter:
    """Filters a directory of .txt files into good / bad quality buckets
    based on KenLM perplexity scores."""

    def __init__(
        self,
        tokenizer_type: str = "sentencepiece",
        num_workers: int | None = None,
    ):
        """
        Args:
            tokenizer_type: ``"sentencepiece"`` or ``"syllable"``.
            num_workers: Number of parallel worker processes.  Defaults to the
                number of CPU cores.
        """
        self._tokenizer_type = tokenizer_type
        self._num_workers = num_workers or multiprocessing.cpu_count()

    def filter_documents(self, input_dir: str, output_dir: str) -> None:
        """Score every ``.txt`` file in *input_dir*, then copy each file to
        either ``good_quality/`` or ``bad_quality/`` inside *output_dir*.

        The threshold is set at the 33rd percentile of perplexity scores so
        that the top one-third of documents are classified as good quality.
        """
        good_quality_path = os.path.join(output_dir, "good_quality")
        bad_quality_path = os.path.join(output_dir, "bad_quality")
        os.makedirs(good_quality_path, exist_ok=True)
        os.makedirs(bad_quality_path, exist_ok=True)

        files_to_process = [
            os.path.join(input_dir, fn)
            for fn in os.listdir(input_dir)
            if fn.endswith(".txt")
        ]

        if not files_to_process:
            print("No .txt files found in the input directory.")
            return

        # -- First pass: calculate perplexity for every file -----------------
        print(
            f"First pass: Calculating perplexity for {len(files_to_process)} files..."
        )

        file_perplexities: dict[str, float] = {}

        with multiprocessing.Pool(
            processes=self._num_workers,
            initializer=_init_worker,
            initargs=(self._tokenizer_type,),
        ) as pool:
            results = pool.imap_unordered(_process_file, files_to_process)

            for result in tqdm(
                results, total=len(files_to_process), desc="Scoring files"
            ):
                if result:
                    filepath, ppl = result
                    file_perplexities[filepath] = ppl

        if not file_perplexities:
            print("No valid documents could be processed.")
            return

        # -- Determine threshold (top third) ---------------------------------
        all_perplexities = list(file_perplexities.values())
        threshold = np.percentile(all_perplexities, 100 / 3)
        print(f"\nGlobal perplexity threshold for top 33.3% (head): {threshold:.4f}")

        # -- Second pass: copy files into buckets ----------------------------
        print("Second pass: Filtering files...")
        good_count = 0
        bad_count = 0

        for input_file, ppl in tqdm(
            file_perplexities.items(), desc="Filtering files"
        ):
            filename = os.path.basename(input_file)
            if ppl <= threshold:
                destination_path = os.path.join(good_quality_path, filename)
                good_count += 1
            else:
                destination_path = os.path.join(bad_quality_path, filename)
                bad_count += 1

            shutil.copy(input_file, destination_path)

        print(f"\nFiltering complete.")
        print(f"  - Saved {good_count} good quality docs to {good_quality_path}")
        print(f"  - Saved {bad_count} bad quality docs to {bad_quality_path}")
