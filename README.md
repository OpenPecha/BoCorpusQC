<h1 align="center">
  <br>
  <a href="https://openpecha.org"><img src="https://avatations.githubusercontent.com/u/82142807?s=400&u=19e108a15566f3a1449bafb03b8dd706a72aebcd&v=4" alt="OpenPecha" width="150"></a>
  <br>
</h1>

# BoCorpusQC: Tibetan Corpus Quality Control

A tool for filtering Tibetan text files based on language model perplexity, separating high-quality documents from low-quality ones.

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

## Programmatic Usage

You can use the main script `kenlm_qc.py` to filter a directory of `.txt` files. The script will process each file, calculate its perplexity, and then sort it into either a `good_quality` or `bad_quality` sub-directory.

### Command-Line Arguments

*   `--input_dir`: **(Required)** The path to the directory containing the `.txt` files you want to filter.
*   `--output_dir`: **(Required)** The path to the directory where the sorted files will be saved.
*   `--num_workers`: (Optional) The number of parallel processes to use for scoring the files. If not specified, it defaults to the total number of CPU cores on your machine.

### Example

```bash
python src/BoCorpusQC/kenlm_qc.py \
    --input_dir /path/to/your/text_files \
    --output_dir /path/to/your/output_folder \
    --num_workers 4
```

This command will:
1.  Process all `.txt` files in `/path/to/your/text_files` using 4 CPU cores.
2.  Create two new folders inside `/path/to/your/output_folder`:
    *   `good_quality`: Contains the top 33.3% of files with the lowest perplexity scores.
    *   `bad_quality`: Contains the remaining 66.7% of files.

## Implementation

This tool evaluates the quality of Tibetan text files using a pre-trained KenLM language model.

1.  **Model Loading**: The script automatically downloads a Tibetan KenLM model (`openpecha/BoKenlm`) and a SentencePiece tokenizer (`openpecha/BoSentencePiece`) from the Hugging Face Hub.
2.  **Perplexity Calculation**: It processes each `.txt` file in the input directory as a single document and calculates its perplexity score. A lower score indicates that the text is more fluent and predictable according to the language model, suggesting higher quality.
3.  **Dynamic Thresholding**: The script calculates a dynamic quality threshold based on the distribution of perplexity scores across all files. It sets the threshold to keep the top one-third of the best-scoring documents. This two-pass approach ensures that the definition of "good quality" is always relative to the specific dataset being processed.
4.  **Parallel Processing**: To speed up computation, the script uses multiprocessing to calculate perplexity scores for multiple files in parallel.
5.  **Output**: Based on the calculated threshold, each file is copied into either the `good_quality` or `bad_quality` subdirectory in your specified output folder.

## Contributing

If you'd like to help out, check out our [contributing guidelines](/CONTRIBUTING.md).

## How to get help

*   File an issue on our GitHub repository.
*   Email us at openpecha[at]gmail.com.
*   Join our [Discord](https://discord.com/invite/7GFpPFSTeA).

## License

This project is licensed under the [MIT License](/LICENSE.md).
