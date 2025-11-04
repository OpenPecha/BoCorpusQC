import argparse
import kenlm
import sentencepiece as spm
import numpy as np
from tqdm import tqdm
import os
from huggingface_hub import hf_hub_download
import shutil
import multiprocessing

# Global variables for worker processes, will be initialized in each worker
kenlm_model = None
sp_model = None


def init_worker():
    """Initializer for multiprocessing pool to load models into each worker process."""
    global kenlm_model, sp_model
    print(f"Initializing worker (PID: {os.getpid()})...")
    # Each worker loads its own copy of the models into its memory space
    kenlm_model, sp_model = load_models()


def process_file(filepath):
    """
    Worker function to calculate perplexity for a single file.
    It relies on the kenlm_model and sp_model being pre-loaded in the worker's
    global scope by the init_worker function.
    """
    global kenlm_model, sp_model
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        if content.strip():
            ppl = calculate_perplexity(content, kenlm_model, sp_model)
            return (filepath, ppl)
    except Exception as e:
        # It's important to catch exceptions in the worker, otherwise the pool can hang
        print(f"Error processing {filepath}: {e}")

    return None


def load_models():
    """Loads the KenLM and SentencePiece models from Hugging Face Hub."""
    print("Downloading models from Hugging Face Hub...")
    arpa_path = hf_hub_download(
        repo_id="openpecha/BoKenlm", filename="bo_kenlm.arpa"
    )
    sp_model_path = hf_hub_download(
        repo_id="openpecha/BoSentencePiece", filename="Bo_sentencepiece.model"
    )

    print("Loading models into memory...")
    kenlm_model = kenlm.Model(arpa_path)
    sp_model = spm.SentencePieceProcessor(model_file=sp_model_path)
    return kenlm_model, sp_model


def calculate_perplexity(doc, kenlm_model, sp_model):
    """Calculates the perplexity of a document."""
    if not doc.strip():
        return float('inf')
        
    log_score = 0
    token_count = 0
    
    # Process paragraph by paragraph as suggested in the paper
    paragraphs = doc.split('\n\n')
    for para in paragraphs:
        if not para.strip():
            continue
        
        # KenLM expects sentences separated by newlines within a paragraph
        sentences = para.strip().split('\n')
        for line in sentences:
            line = line.strip()
            if not line:
                continue
            
            tokens = " ".join(sp_model.encode_as_pieces(line))
            
            # score() returns the log10 probability of the sentence.
            log_score += kenlm_model.score(tokens, bos=True, eos=True)
            token_count += len(tokens.split()) + 1 # +1 for </s>

    if token_count == 0:
        return float('inf')

    # Perplexity = 10^(-log10(p) / N)
    perplexity = 10 ** (-log_score / token_count)
    return perplexity

def filter_documents(input_dir, output_dir):
    """
    Filters documents in an input directory based on perplexity and saves them
    to good_quality and bad_quality subdirectories in the output directory.
    Each file in the input directory is considered a single document.
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

    # First pass: calculate perplexity for each file using a pool of worker processes
    print(f"First pass: Calculating perplexity for {len(files_to_process)} files...")

    file_perplexities = {}
    with multiprocessing.Pool(initializer=init_worker) as pool:
        # Use imap_unordered for efficiency, as the order of processing doesn't matter
        results = pool.imap_unordered(process_file, files_to_process)

        # Wrap with tqdm to show a progress bar
        for result in tqdm(results, total=len(files_to_process), desc="Scoring files"):
            if result:
                filepath, ppl = result
                file_perplexities[filepath] = ppl

    if not file_perplexities:
        print("No valid documents could be processed.")
        return

    # Determine the perplexity threshold for the top third (head)
    all_perplexities = list(file_perplexities.values())
    threshold = np.percentile(all_perplexities, 100 / 3)
    print(f"\nGlobal perplexity threshold for top 33.3% (head): {threshold:.4f}")

    # Second pass: filter files based on the threshold
    print("Second pass: Filtering files...")
    good_count = 0
    bad_count = 0
    for input_file, ppl in tqdm(file_perplexities.items(), desc="Filtering files"):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter documents based on KenLM perplexity."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the documents to be filtered.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the filtered documents.",
    )
    args = parser.parse_args()

    filter_documents(args.input_dir, args.output_dir)
