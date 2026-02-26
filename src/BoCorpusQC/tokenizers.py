import re
from abc import ABC, abstractmethod

import sentencepiece as spm
from huggingface_hub import hf_hub_download


class BaseTokenizer(ABC):
    """Abstract base class for all tokenizers."""

    @abstractmethod
    def tokenize(self, line: str) -> str:
        """Tokenize a single line of text.

        Args:
            line: A single line of text to tokenize.

        Returns:
            A string of space-separated tokens.
        """


class SentencePieceTokenizer(BaseTokenizer):
    """Tokenizer that uses a SentencePiece model downloaded from Hugging Face Hub."""

    HF_REPO_ID = "openpecha/BoSentencePiece"
    HF_FILENAME = "sentencepiece.model"

    def __init__(self):
        print("Downloading SentencePiece model from Hugging Face Hub...")
        sp_model_path = hf_hub_download(
            repo_id=self.HF_REPO_ID, filename=self.HF_FILENAME
        )
        print("Loading SentencePiece model into memory...")
        self._sp = spm.SentencePieceProcessor(model_file=sp_model_path)

    def tokenize(self, line: str) -> str:
        """Tokenize a line using the SentencePiece model.

        Args:
            line: A single line of text to tokenize.

        Returns:
            A string of space-separated sub-word tokens.
        """
        return " ".join(self._sp.encode_as_pieces(line))


class SyllableTokenizer(BaseTokenizer):
    """Tokenizer that splits Tibetan text on the tsek (་) and shad (།) characters."""

    TSEK = "་"
    SHAD = "།"
    _DELIMITER_RE = re.compile(r"([་།])")

    def tokenize(self, line: str) -> str:
        """Tokenize a line by splitting on Tibetan tsek and shad characters.

        Each delimiter is kept attached to the syllable it follows.

        Args:
            line: A single line of Tibetan text to tokenize.

        Returns:
            A string of space-separated syllables (with delimiters attached).
        """
        parts = self._DELIMITER_RE.split(line)
        syllables: list[str] = []
        i = 0
        while i < len(parts):
            text = parts[i].strip()
            delimiter = parts[i + 1] if i + 1 < len(parts) else ""
            if text:
                syllables.append(text + delimiter)
            elif delimiter and syllables:
                # Orphaned delimiter (e.g. consecutive ་།) — attach to previous syllable
                syllables[-1] += delimiter
            i += 2
        return " ".join(syllables)
