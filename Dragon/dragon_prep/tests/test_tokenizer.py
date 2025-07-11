from dragon_prep.ner import ner_tokenizer


def test_tokenizer():
    s = 'een "twee"->drie "vier" vijf?->zes zeven? 8='
    tokenizer = ner_tokenizer()
    tokenized_text = tokenizer(s)
    text_parts = [token.text for token in tokenized_text]

    print(text_parts)
    assert text_parts == ['een', '"', 'twee', '"', '-', '>', 'drie', '"', 'vier', '"', 'vijf', '?', '-', '>', 'zes', 'zeven', '?', '8=']


if __name__ == "__main__":
    test_tokenizer()
