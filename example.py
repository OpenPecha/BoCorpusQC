"""
Example: Calculate perplexity of a Tibetan text using both tokenizer options.
"""

import time
from pathlib import Path

from BoCorpusQC import PerplexityCalculator


def calculate_sentencepiece_perplexity(text: str) -> float:
    """Calculate perplexity using the SentencePiece tokenizer.

    Models are downloaded automatically from Hugging Face Hub.

    Args:
        text: Tibetan text to score.

    Returns:
        The perplexity score.
    """
    calculator = PerplexityCalculator.from_sentencepiece()
    return calculator.calculate_perplexity(text)


def calculate_syllable_perplexity(text: str) -> float:
    """Calculate perplexity using the syllable-level tokenizer.

    KenLM model is downloaded automatically from Hugging Face Hub.

    Args:
        text: Tibetan text to score.

    Returns:
        The perplexity score.
    """
    calculator = PerplexityCalculator.from_syllable()
    return calculator.calculate_perplexity(text)


def main():
    tibetan_text = Path("./data/doc7_c.txt").read_text(encoding="utf-8")

    # 1. SentencePiece perplexity
    print("=" * 60)
    print("Perplexity — SentencePiece tokenizer")
    print("=" * 60)
    start_sp = time.perf_counter()
    ppl_sp = calculate_sentencepiece_perplexity(tibetan_text)
    time_sp = time.perf_counter() - start_sp
    print(f"  Perplexity: {ppl_sp:.4f}")
    print(f"  Time:       {time_sp:.4f}s")
    print()

    # 2. Syllable perplexity
    print("=" * 60)
    print("Perplexity — Syllable tokenizer")
    print("=" * 60)
    start_syl = time.perf_counter()
    ppl_syl = calculate_syllable_perplexity(tibetan_text)
    time_syl = time.perf_counter() - start_syl
    print(f"  Perplexity: {ppl_syl:.4f}")
    print(f"  Time:       {time_syl:.4f}s")
    print()

    # 3. Compare
    print("=" * 60)
    print("Comparison")
    print("=" * 60)
    print(f"  SentencePiece perplexity: {ppl_sp:.4f}  ({time_sp:.4f}s)")
    print(f"  Syllable perplexity:      {ppl_syl:.4f}  ({time_syl:.4f}s)")


if __name__ == "__main__":
    main()
