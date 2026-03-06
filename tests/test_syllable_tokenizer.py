import pytest

from BoCorpusQC.tokenizers import SyllableTokenizer


@pytest.fixture
def tokenizer():
    return SyllableTokenizer()


def test_basic_tibetan_text(tokenizer):
    """Syllables separated by tsek are split correctly."""
    line = "བཀྲ་ཤིས་བདེ་ལེགས"
    result = tokenizer.tokenize(line)
    # Each syllable (with its trailing tsek) should be space-separated
    assert "བཀྲ" in result
    assert "ཤིས" in result
    assert "བདེ" in result
    assert "ལེགས" in result


def test_empty_string(tokenizer):
    """An empty string returns an empty string."""
    result = tokenizer.tokenize("")
    assert result == ""


def test_shad_at_end(tokenizer):
    """Shad at end of sentence is preserved in the output."""
    line = "བཀྲ་ཤིས་བདེ་ལེགས།"
    result = tokenizer.tokenize(line)
    assert "།" in result


def test_multiword_sentence(tokenizer):
    """A longer Tibetan sentence produces multiple tokens."""
    line = "བོད་སྐད་ཀྱི་ཚིག་གྲུབ་འདི་ཡིན།"
    result = tokenizer.tokenize(line)
    tokens = result.split()
    assert len(tokens) >= 2


def test_return_type_is_string(tokenizer):
    """The tokenize method returns a string (space-separated tokens)."""
    line = "བཀྲ་ཤིས་བདེ་ལེགས"
    result = tokenizer.tokenize(line)
    assert isinstance(result, str)
