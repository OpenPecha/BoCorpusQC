"""Command-line interface for BoCorpusQC."""

import argparse
import multiprocessing

from BoCorpusQC.filter import DocumentFilter


def main():
    parser = argparse.ArgumentParser(
        description="Filter Tibetan documents based on KenLM perplexity."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing the .txt files to be filtered.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the filtered documents.",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        choices=["sentencepiece", "syllable"],
        default="sentencepiece",
        help="Tokenizer to use: 'sentencepiece' (default) or 'syllable'.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of parallel processes to use. Defaults to the number of CPU cores.",
    )

    args = parser.parse_args()

    doc_filter = DocumentFilter(
        tokenizer_type=args.tokenizer_type,
        num_workers=args.num_workers,
    )
    doc_filter.filter_documents(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
