from src.rag.infrastructure._sentence_splitter import split_sentences


def test_split_sentences_splits_on_terminal_punctuation():
    sentences = split_sentences("First sentence. Second sentence! Third sentence?")
    assert sentences == ["First sentence.", "Second sentence!", "Third sentence?"]


def test_split_sentences_handles_a_single_sentence():
    assert split_sentences("Just one sentence.") == ["Just one sentence."]


def test_split_sentences_handles_empty_text():
    assert split_sentences("") == []


def test_split_sentences_does_not_split_on_a_decimal_number():
    sentences = split_sentences("The value is 3.14 and it matters. Next sentence.")
    assert sentences == ["The value is 3.14 and it matters.", "Next sentence."]
