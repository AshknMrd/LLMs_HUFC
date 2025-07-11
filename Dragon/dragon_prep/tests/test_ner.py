import pytest

from dragon_prep.ner import (fix_sequence_labels_after_anon, ner_tokenizer,
                             reconstruct_text)


def test_tokenizer_reconstruction():
    # make some sample text with special characters and newlines etc.
    text = """
    Hello, World! 🌍✨

    This is a test document with special characters:

    - Bullet points:
    - ✓ Checkmark
    - ✗ Cross
    - → Arrow

    1. Numbers with symbols:
    - 1️⃣ First
    - 2️⃣ Second
    - 3️⃣ Third

    💡 Tips:  
        - Use `Ctrl + C` to copy.  
        - Use `Ctrl + V` to paste.  

    Mathematical symbols:  
        ∑ (Summation), ∫ (Integral), √ (Square root), π (Pi)

    Line break example:  
    First line...  
    Second line...  

    Tab example:  
    \tTabbed text starts here.

    Quotes:  
    "To be, or not to be, that is the question." — *William Shakespeare*

    End of sample.  
    Thank you! 🙏
    """

    # Tokenize the text
    tokenizer = ner_tokenizer()
    tokenized_text = tokenizer(text)

    # Reconstruct the text
    reconstructed_text = reconstruct_text(tokenized_text)

    # Check if the reconstructed text is the same as the original text
    assert reconstructed_text == text


@pytest.mark.parametrize(
    "text_orig, text_anon, labels_orig, expected_labels_anon",
    [
        (
            # Test case 1: No changes in text and labels
            "This code was written by Joeran Bosma.",
            "This code was written by Joeran Bosma.",
            [(25, 37, "<PERSON>")],
            [(25, 37, "<PERSON>")],
        ),
        (
            # Test case 2: Insertions before and after the entity
            "This code was written by Joeran Bosma.",
            "This beautiful code was written by Joeran Bosma. Cool right?",
            [(25, 37, "<PERSON>")],
            [(35, 47, "<PERSON>")],
        ),
        (
            # Test case 3: Insertions within the entity
            "This code was written by Joeran Bosma.",
            "This code was written by Joeran S. Bosma.",
            [(25, 37, "<PERSON>")],
            [(25, 40, "<PERSON>")],
        ),
        (
            # Test case 4: Replacement of the entity
            "This code was written by GitHub Copilot.",
            "This code was written by Joeran Bosma.",
            [(25, 39, "<PERSON>")],
            [(25, 37, "<PERSON>")],
        ),
        (
            # Test case 5: Replacement of the entity with function description
            "This code was written by GitHub Copilot.",
            "This code was written by Joeran Bosma, PhD candidate.",
            [(25, 39, "<PERSON>")],
            [(25, 52, "<PERSON>")],
        ),
        (
            # Test case 6: Replacement of the entity with function description, where the end of the name overlaps
            "This code was written by J. Bosma.",
            "This code was written by Joeran Bosma, PhD candidate.",
            [(25, 33, "<PERSON>")],
            [(25, 52, "<PERSON>")],
        ),
        (
            # Test case 7: Replacement of the entity with new insertion on the left
            "This code was written by J. Bosma.",
            "This code was written by PhD candidate Joeran Bosma.",
            [(25, 33, "<PERSON>")],
            [(25, 51, "<PERSON>")],
        ),
        (
            # Test case 8: Replacement of the entity with new insertion on the left, but separate from the entity
            "This code was written by J. Bosma.",
            "This code was written as a PhD candidate, specifically by J. Bosma.",
            [(25, 33, "<PERSON>")],
            [(58, 66, "<PERSON>")],
        ),
        (
            # Test case 9: Replacement with partial overlap
            "Code changed in februari 2028",
            "Code changed in september 2018",
            [(16, 29, "<DATE>")],
            [(16, 30, "<DATE>")],
        ),
        (
            # Test case 10: Replacement of the entity with function description (edge case, with a dot in the function description)
            "This code was written by GitHub Copilot. Do you like the tests?",
            "This code was written by Joeran Bosma, M.Sc.. Do you like the tests?",
            [(25, 39, "<PERSON>")],
            [(25, 44, "<PERSON>")],
        ),
        # (
        #     # Test case X: Replacement of the entity with function description (edge case, with a dot in the function description)
        #     # this one still fails, but it should not happen in practice that there is no text after the entitties like this.
        #     # any suggestions on how to improve this are welcome!
        #     "This code was written by GitHub Copilot.",
        #     "This code was written by Joeran Bosma, M.Sc..",
        #     [(25, 39, "<PERSON>")],
        #     [(25, 44, "<PERSON>")],
        # ),
    ],
)
def test_fix_sequence_labels_after_anon(text_orig, labels_orig, text_anon, expected_labels_anon):
    result = fix_sequence_labels_after_anon(
        text_orig=text_orig,
        text_anon=text_anon,
        labels_orig=labels_orig,
        strict=False,
        expand_on_neighboring_insertions=True
    )
    for start, stop, _ in labels_orig:
        print(f"Selected before: '{text_orig[start:stop]}'")
    for start, stop, _ in result:
        print(f"Selected after: '{text_anon[start:stop]}'")
    assert result == expected_labels_anon


if __name__ == "__main__":
    pytest.main(['-v', '-k', 'test_tokenizer_reconstruction'])
