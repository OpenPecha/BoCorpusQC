from __future__ import annotations

import kenlm
from huggingface_hub import hf_hub_download

from BoCorpusQC.tokenizers import BaseTokenizer, SentencePieceTokenizer, SyllableTokenizer

HF_KENLM_SP_REPO_ID = "openpecha/BoKenlm-sp"
HF_KENLM_SYL_REPO_ID = "openpecha/BoKenlm-syl"
HF_KENLM_SP_FILENAME = "BoKenlm-sp.arpa"
HF_KENLM_SYL_FILENAME = "BoKenlm-syl.arpa"


# ---------------------------------------------------------------------------
# Standalone model loaders
# ---------------------------------------------------------------------------


def load_kenlm_model(model_path: str) -> kenlm.Model:
    """Load a KenLM model from a local file.

    Args:
        model_path: Path to the KenLM model file (.arpa or .binary).

    Returns:
        A loaded ``kenlm.Model`` instance.
    """
    return kenlm.Model(model_path)


def load_sp_kenlm_model() -> kenlm.Model:
    """Download and load the SentencePiece-level KenLM model from Hugging Face Hub.

    Returns:
        A loaded ``kenlm.Model`` instance.
    """
    arpa_path = hf_hub_download(
        repo_id=HF_KENLM_SP_REPO_ID, filename=HF_KENLM_SP_FILENAME
    )
    return kenlm.Model(arpa_path)


def load_syl_kenlm_model() -> kenlm.Model:
    """Download and load the syllable-level KenLM model from Hugging Face Hub.

    Returns:
        A loaded ``kenlm.Model`` instance.
    """
    arpa_path = hf_hub_download(
        repo_id=HF_KENLM_SYL_REPO_ID, filename=HF_KENLM_SYL_FILENAME
    )
    return kenlm.Model(arpa_path)


# ---------------------------------------------------------------------------
# Standalone perplexity calculation
# ---------------------------------------------------------------------------


def calculate_perplexity(
    kenlm_model: kenlm.Model,
    tokenizer: BaseTokenizer,
    doc: str,
) -> float:
    """Calculate the perplexity of a document.

    Args:
        kenlm_model: A pre-loaded ``kenlm.Model``.
        tokenizer: A tokenizer instance implementing ``BaseTokenizer``.
        doc: The full text of a document (may contain multiple lines).

    Returns:
        The perplexity score.  Lower is better (more fluent).
        Returns ``float('inf')`` for empty documents.
    """
    if not doc.strip():
        return float("inf")

    log_score = 0.0
    token_count = 0

    for line in doc.split("\n"):
        line = line.strip()
        if not line:
            continue

        tokens = tokenizer.tokenize(line)

        log_score += kenlm_model.score(tokens, bos=True, eos=True)
        token_count += len(tokens.split()) + 1  # +1 for </s>

    if token_count == 0:
        return float("inf")

    return 10 ** (-log_score / token_count)


# ---------------------------------------------------------------------------
# Class-based API (convenience wrapper)
# ---------------------------------------------------------------------------


class PerplexityCalculator:
    """Calculates perplexity of Tibetan text using a KenLM model and a tokenizer."""

    def __init__(self, kenlm_model: kenlm.Model, tokenizer: BaseTokenizer):
        """Initialize the calculator with pre-loaded components.

        Args:
            kenlm_model: A pre-loaded ``kenlm.Model``.
            tokenizer: A tokenizer instance implementing ``BaseTokenizer``.
        """
        self._kenlm = kenlm_model
        self._tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls, kenlm_model_path: str, tokenizer: BaseTokenizer
    ) -> PerplexityCalculator:
        """Create a calculator by loading a KenLM model from a file path.

        Args:
            kenlm_model_path: Path to the KenLM model file (.arpa or .binary).
            tokenizer: A tokenizer instance implementing ``BaseTokenizer``.

        Returns:
            A configured ``PerplexityCalculator`` instance.
        """
        model = load_kenlm_model(kenlm_model_path)
        return cls(kenlm_model=model, tokenizer=tokenizer)

    @classmethod
    def from_sentencepiece(cls) -> PerplexityCalculator:
        """Create a calculator using models downloaded from Hugging Face Hub.

        Downloads both the KenLM language model and the SentencePiece tokenizer
        from the OpenPecha repositories on Hugging Face Hub.

        Returns:
            A configured ``PerplexityCalculator`` instance.
        """
        model = load_sp_kenlm_model()
        tokenizer = SentencePieceTokenizer()
        return cls(kenlm_model=model, tokenizer=tokenizer)

    @classmethod
    def from_syllable(cls) -> PerplexityCalculator:
        """Create a calculator using syllable-level tokenization.

        Downloads the syllable-level KenLM model from the OpenPecha
        repository on Hugging Face Hub.

        Returns:
            A configured ``PerplexityCalculator`` instance.
        """
        model = load_syl_kenlm_model()
        tokenizer = SyllableTokenizer()
        return cls(kenlm_model=model, tokenizer=tokenizer)

    def calculate_perplexity(self, doc: str) -> float:
        """Calculate the perplexity of a document.

        Args:
            doc: The full text of a document (may contain multiple lines).

        Returns:
            The perplexity score.  Lower is better (more fluent).
            Returns ``float('inf')`` for empty documents.
        """
        return calculate_perplexity(self._kenlm, self._tokenizer, doc)
