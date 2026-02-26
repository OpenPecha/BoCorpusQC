import pytest

from BoCorpusQC.tokenizers import SyllableTokenizer


@pytest.fixture
def tokenizer():
    return SyllableTokenizer()


def test_basic_tibetan_text(tokenizer):
    """Syllables separated by tsek are split correctly, tsek stays attached."""
    line = "བཀྲ་ཤིས་བདེ་ལེགས"
    result = tokenizer.tokenize(line)
    assert result == "བཀྲ་ ཤིས་ བདེ་ ལེགས"


def test_single_syllable(tokenizer):
    """A single syllable with no tsek returns itself."""
    line = "བཀྲ"
    result = tokenizer.tokenize(line)
    assert result == "བཀྲ"


def test_trailing_tsek(tokenizer):
    """Trailing tsek stays attached to its syllable."""
    line = "བཀྲ་ཤིས་"
    result = tokenizer.tokenize(line)
    assert result == "བཀྲ་ ཤིས་"


def test_leading_tsek(tokenizer):
    """Leading tsek should not produce an empty leading token."""
    line = "་བཀྲ་ཤིས"
    result = tokenizer.tokenize(line)
    assert result == "བཀྲ་ ཤིས"


def test_multiple_consecutive_tseks(tokenizer):
    """Multiple consecutive tseks should not produce empty tokens."""
    line = "བཀྲ་་་ཤིས"
    result = tokenizer.tokenize(line)
    assert result == "བཀྲ་ ཤིས"


def test_empty_string(tokenizer):
    """An empty string returns an empty string."""
    result = tokenizer.tokenize("")
    assert result == ""


def test_whitespace_only(tokenizer):
    """A whitespace-only string returns an empty string."""
    result = tokenizer.tokenize("   ")
    assert result == ""


def test_shad_at_end(tokenizer):
    """Shad at end of sentence stays attached to the last syllable."""
    line = "བཀྲ་ཤིས་བདེ་ལེགས།"
    result = tokenizer.tokenize(line)
    assert result == "བཀྲ་ ཤིས་ བདེ་ ལེགས།"


def test_shad_between_sentences(tokenizer):
    """Shad separates two clauses; it attaches to the preceding syllable."""
    line = "བཀྲ་ཤིས།བདེ་ལེགས"
    result = tokenizer.tokenize(line)
    assert result == "བཀྲ་ ཤིས། བདེ་ ལེགས"


def test_tsek_then_shad(tokenizer):
    """Tsek followed by shad (་།) both attach to the preceding syllable."""
    line = "བཀྲ་ཤིས་།བདེ་ལེགས"
    result = tokenizer.tokenize(line)
    assert result == "བཀྲ་ ཤིས་། བདེ་ ལེགས"


def test_double_shad(tokenizer):
    """Double shad (།།) stays attached to the preceding syllable."""
    line = "ལེགས།།"
    result = tokenizer.tokenize(line)
    assert result == "ལེགས།།"


def test_spaces_around_syllables_are_stripped(tokenizer):
    """Extra whitespace around syllables is stripped, delimiters preserved."""
    line = " བཀྲ ་ ཤིས ་ བདེ "
    result = tokenizer.tokenize(line)
    assert result == "བཀྲ་ ཤིས་ བདེ"


def test_tsek_only(tokenizer):
    """A string of only tseks returns an empty string."""
    line = "་་་"
    result = tokenizer.tokenize(line)
    assert result == ""


def test_shad_only(tokenizer):
    """A string of only shads returns an empty string."""
    line = "།།།"
    result = tokenizer.tokenize(line)
    assert result == ""
