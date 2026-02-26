from __future__ import annotations

import kenlm
from huggingface_hub import hf_hub_download

from BoCorpusQC.tokenizers import BaseTokenizer, SentencePieceTokenizer, SyllableTokenizer


class PerplexityCalculator:
    """Calculates perplexity of Tibetan text using a KenLM model and a tokenizer."""

    HF_KENLM_SP_REPO_ID = "openpecha/BoKenlm-sp"
    HF_KENLM_SYL_REPO_ID = "openpecha/BoKenlm-syl"
    HF_KENLM_SP_FILENAME = "BoKenlm-sp.arpa"
    HF_KENLM_SYL_FILENAME = "BoKenlm-syl.arpa"  # TODO: check if this is correct

    def __init__(self, kenlm_model_path: str, tokenizer: BaseTokenizer):
        """Initialize the calculator with a KenLM model and a tokenizer.

        Args:
            kenlm_model_path: Path to the KenLM model file (.arpa or .binary).
            tokenizer: A tokenizer instance implementing BaseTokenizer.
        """
        print("Loading KenLM model into memory...")
        self._kenlm = kenlm.Model(kenlm_model_path)
        self._tokenizer = tokenizer

    # -- Convenient factory class methods -----------------------------------

    @classmethod
    def from_sentencepiece(cls) -> PerplexityCalculator:
        """Create a calculator using models downloaded from Hugging Face Hub.

        Downloads both the KenLM language model and the SentencePiece tokenizer
        from the OpenPecha repositories on Hugging Face Hub.

        Returns:
            A configured PerplexityCalculator instance.
        """
        print("Downloading KenLM model from Hugging Face Hub...")
        arpa_path = hf_hub_download(
            repo_id=cls.HF_KENLM_SP_REPO_ID, filename=cls.HF_KENLM_SP_FILENAME
        )
        tokenizer = SentencePieceTokenizer()
        return cls(kenlm_model_path=arpa_path, tokenizer=tokenizer)

    @classmethod
    def from_syllable(cls) -> PerplexityCalculator:
        """Create a calculator using syllable-level tokenization.

        Downloads the syllable-level KenLM model from the OpenPecha
        repository (``openpecha/BoKenlm-syl``) on Hugging Face Hub.

        Returns:
            A configured PerplexityCalculator instance.
        """
        print("Downloading syllable KenLM model from Hugging Face Hub...")
        arpa_path = hf_hub_download(
            repo_id=cls.HF_KENLM_SYL_REPO_ID, filename=cls.HF_KENLM_SYL_FILENAME
        )
        tokenizer = SyllableTokenizer()
        return cls(kenlm_model_path=arpa_path, tokenizer=tokenizer)

    # -- Core scoring method ------------------------------------------------

    def calculate_perplexity(self, doc: str) -> float:
        """Calculate the perplexity of a document.

        Args:
            doc: The full text of a document (may contain multiple lines).

        Returns:
            The perplexity score.  Lower is better (more fluent).
            Returns float('inf') for empty documents.
        """
        if not doc.strip():
            return float("inf")

        log_score = 0
        token_count = 0

        for line in doc.split("\n"):
            line = line.strip()
            if not line:
                continue

            tokens = self._tokenizer.tokenize(line)

            # score() returns the log10 probability of the sentence.
            log_score += self._kenlm.score(tokens, bos=True, eos=True)
            token_count += len(tokens.split()) + 1  # +1 for </s>

        if token_count == 0:
            return float("inf")

        # Perplexity = 10^(-log10(p) / N)
        perplexity = 10 ** (-log_score / token_count)
        return perplexity
