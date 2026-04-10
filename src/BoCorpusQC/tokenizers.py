from abc import ABC, abstractmethod

import sentencepiece as spm
from botok_rs import SimpleTokenizer as BotokTokenizer
from botok.utils.corpus_normalization import normalize_for_perplexity
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
    """Tokenizes Tibetan text into syllables using botok-rs.

    Example:
        >>> tok = SyllableTokenizer()
        >>> tok.tokenize("བོད་སྐད་ཀྱི་ཚིག་གྲུབ་འདི་ཡིན།")
        'བོད་ སྐད་ ཀྱི་ ཚིག་ གྲུབ་ འདི་ ཡིན།'
    """

    def tokenize(self, line: str) -> str:
        """Tokenize a single line of Tibetan text into syllables.

        Uses botok-rs ``SimpleTokenizer`` under the hood.

        Args:
            line: A single line of Tibetan text to tokenize.

        Returns:
            A string of space-separated syllable tokens.
        """
        tokens = BotokTokenizer.tokenize(line)
        return " ".join(token.text for token in tokens if token.text.strip())


class NormalizedSyllableTokenizer(BaseTokenizer):
    """Tokenizes Tibetan text into normalized syllables using botok.

    Example:
        >>> tok = NormalizedSyllableTokenizer()
        >>> tok.tokenize("བོད་སྐད་ཀྱི་ཚིག་གྲུབ་འདི་ཡིན།")
        ['བོད་', 'སྐད་', 'ཀྱི་', 'ཚིག་', 'གྲུབ་', 'འདི་', 'ཡིན།']
    """

    def tokenize(self, text: str) -> list[str]:
        """Tokenize a single line of Tibetan text into syllables.

        Args:
            text: A line of Tibetan text to tokenize.

        Returns:
            A list of syllable strings.
        """
        tokenized_text = normalize_for_perplexity(text=text, space_sskt=True)
        tokens = tokenized_text.split(" ")
        return tokens
