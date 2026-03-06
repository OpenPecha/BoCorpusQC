<h1 align="center">
  <br>
  <a href="https://openpecha.org"><img src="https://avatations.githubusercontent.com/u/82142807?s=400&u=19e108a15566f3a1449bafb03b8dd706a72aebcd&v=4" alt="OpenPecha" width="150"></a>
  <br>
</h1>

# BoCorpusQC: Tibetan Corpus Quality Control

A tool for filtering Tibetan text files based on language model perplexity, separating high-quality documents from low-quality ones. Two tokenization strategies are supported:

*   **SentencePiece** (default) — uses a sub-word model downloaded from Hugging Face Hub.
*   **Syllable** — splits text on the Tibetan tsek character (`་`) and uses a KenLM model (`openpecha/BoKenlm-syl`) downloaded from Hugging Face Hub.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-repo/BoCorpusQC.git
    cd BoCorpusQC
    ```

2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Command-Line Usage

Use `cli.py` to filter a directory of `.txt` files. Each file is scored by perplexity and sorted into `good_quality` or `bad_quality` sub-directories.

### Command-Line Arguments

*   `--input_dir`: **(Required)** The path to the directory containing the `.txt` files you want to filter.
*   `--output_dir`: **(Required)** The path to the directory where the sorted files will be saved.
*   `--tokenizer_type`: (Optional) `sentencepiece` (default) or `syllable`.
*   `--num_workers`: (Optional) The number of parallel processes to use for scoring the files. Defaults to the total number of CPU cores on your machine.

### Example — SentencePiece (default)

```bash
python -m BoCorpusQC.cli \
    --input_dir /path/to/your/text_files \
    --output_dir /path/to/your/output_folder \
    --num_workers 4
```

This command will:
1.  Process all `.txt` files in `/path/to/your/text_files` using 4 CPU cores.
2.  Create two new folders inside `/path/to/your/output_folder`:
    *   `good_quality`: Contains the top 33.3% of files with the lowest perplexity scores.
    *   `bad_quality`: Contains the remaining 66.7% of files.

### Example — Syllable tokenization

```bash
python -m BoCorpusQC.cli \
    --input_dir /path/to/your/text_files \
    --output_dir /path/to/your/output_folder \
    --tokenizer_type syllable \
    --num_workers 4
```

## Programmatic Usage

### Calculate Perplexity for a Text String (SentencePiece)

```python
from BoCorpusQC import PerplexityCalculator

# Downloads KenLM + SentencePiece models from Hugging Face Hub automatically
calculator = PerplexityCalculator.from_sentencepiece()

text = "བཀྲ་ཤིས་བདེ་ལེགས། ཁམས་བཟང་ངམ།"
ppl = calculator.calculate_perplexity(text)
print(f"Perplexity: {ppl:.4f}")
```

### Calculate Perplexity for a Text String (Syllable)

```python
from BoCorpusQC import PerplexityCalculator

# Downloads syllable-level KenLM model from Hugging Face Hub automatically
calculator = PerplexityCalculator.from_syllable()

text = "བཀྲ་ཤིས་བདེ་ལེགས། ཁམས་བཟང་ངམ།"
ppl = calculator.calculate_perplexity(text)
print(f"Perplexity: {ppl:.4f}")
```

### Filter Documents Programmatically

```python
from BoCorpusQC import DocumentFilter

# Using the default SentencePiece tokenizer
doc_filter = DocumentFilter(tokenizer_type="sentencepiece", num_workers=4)
doc_filter.filter_documents("/path/to/input", "/path/to/output")

# Or using syllable-level tokenization
doc_filter = DocumentFilter(
    tokenizer_type="syllable",
    num_workers=4,
)
doc_filter.filter_documents("/path/to/input", "/path/to/output")
```

A **lower** perplexity score indicates that the text is more fluent and predictable according to the language model, suggesting higher quality.

## Implementation

This tool evaluates the quality of Tibetan text files using a pre-trained KenLM language model.

1.  **Tokenization**: Two strategies are available. *SentencePiece* downloads a sub-word tokenizer (`openpecha/BoSentencePiece`) from Hugging Face Hub. *Syllable* splits text on the Tibetan tsek character (`་`).
2.  **Model Loading**: For SentencePiece mode the KenLM model (`openpecha/BoKenlm`) is automatically downloaded from Hugging Face Hub. For syllable mode the KenLM model (`openpecha/BoKenlm-syl`) is automatically downloaded from Hugging Face Hub.
3.  **Perplexity Calculation**: Each `.txt` file in the input directory is treated as a single document and scored. A lower score means more fluent, higher-quality text.
4.  **Dynamic Thresholding**: A quality threshold is computed at the 33rd percentile of all perplexity scores, keeping the top one-third of documents as "good quality".
5.  **Parallel Processing**: Multiprocessing is used to score many files in parallel.
6.  **Output**: Each file is copied into either the `good_quality` or `bad_quality` subdirectory in the specified output folder.

## Contributing

If you'd like to help out, check out our [contributing guidelines](/CONTRIBUTING.md).

## How to get help

*   File an issue on our GitHub repository.
*   Email us at openpecha[at]gmail.com.
*   Join our [Discord](https://discord.com/invite/7GFpPFSTeA).

## Acknowledgements

Developed as part of [The BDRC E-Text Corpus project](https://www.bdrc.io/blog/2026/02/28/bdrc-launches-major-initiative-to-build-open-buddhist-datasets-for-ai/).

## License

This project is licensed under the [MIT License](/LICENSE.md).
