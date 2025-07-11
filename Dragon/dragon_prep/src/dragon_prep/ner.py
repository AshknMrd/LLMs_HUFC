#  Copyright 2022 Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import difflib
import json
import warnings
from copy import deepcopy
from typing import Any

import pandas as pd
import spacy
from spacy.lang.nl import Dutch
from spacy.tokenizer import Tokenizer
from spacy.training import offsets_to_biluo_tags
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def biluo_to_bio_tags(tags: list[str], strict: bool = True) -> list[str]:
    """BILUO is an extended version of the BIO scheme.
    B - beginning, I - inside, L - last, U - unigram, O - outside.

    U and L can be reduced to B and I in the BIO scheme.
    Tags always have form of <identifier>-<class> (where identifier is B or I) or "O"
    (when the token does not have an entity class).

    E.g. B-PERSON, I-ORG, O
    """
    new_tags: list[str] = []
    num_tokens_skipped = 0
    for t in tags:
        if t == "-":
            num_tokens_skipped += 1
            # spacy inserts "-" when offsets don't line up with tokens after tokenization
            new_tags.append("O")
        elif t.startswith("U"):
            # replace with B-<class>
            new_tags.append("B" + t[1:])
        elif t.startswith("L"):
            # replace with I-<class>
            new_tags.append("I" + t[1:])
        else:
            # this case handles "O"
            new_tags.append(t)

    if num_tokens_skipped > 0:
        tqdm.write(f"{num_tokens_skipped} misaligned tokens replaced with O")

    if strict and num_tokens_skipped > 0:
        raise ValueError("Strict mode is enabled, but misaligned tokens were found!")

    return new_tags


def ner_tokenizer() -> Tokenizer:
    """Define the tokenizer to use for splitting text into tokens for named entity recognition."""
    # Create a custom tokenizer that's not tied to a specific vocabulary
    vocab = spacy.vocab.Vocab()

    # Add custom prefixes and suffixes for splitting text into tokens
    split_chars = r'[:;,_\-\.\/\\\+\(\)~<>*"\?]'
    prefixes = Dutch.Defaults.prefixes + [split_chars]
    suffixes = [p for p in Dutch.Defaults.suffixes if p != r'\.\.+'] + [split_chars]
    infixes = [p for p in Dutch.Defaults.infixes if p != r'\.\.+'] + [
        split_chars,
        r'(?<=\d)(?=[a-zA-Z])',  # number followed by letter
        r'(?<=[a-zA-Z])(?=\d)',  # letter followed by number
    ]

    # Create a custom tokenizer with the updated prefix and suffix regex
    tokenizer = Tokenizer(
        vocab,
        prefix_search=spacy.util.compile_prefix_regex(prefixes).search,
        suffix_search=spacy.util.compile_suffix_regex(suffixes).search,
        infix_finditer=spacy.util.compile_infix_regex(infixes).finditer,
    )

    return tokenizer


def doccano_to_bio_tags(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Converts doccano tags to BIO tags."""
    tokenizer = ner_tokenizer()
    processed_docs = []
    for row in tqdm(data, desc="Converting offsets to BIO tags"):
        text = row["text"]
        labels = row["labels"] if "labels" in row else row["label"]

        tokenized_text = tokenizer(text)
        tags = offsets_to_biluo_tags(
            tokenized_text, [(*span,) for span in labels]
        )

        if "-" in tags:
            print("Misaligned text:", [t.text for t, tag in zip(tokenized_text, tags) if tag == "-"])
            print("Selected text:", [text[start:stop] for start, stop, _ in labels])
            print(f"ID: {row.get('id', 'unknown')}, UID: {row.get('uid', 'unknown')}")
            print(f"Report: {text}")
            print("Have misaligned tokens!")

        # convert BILUO to BIO
        tags = biluo_to_bio_tags(tags, strict=True)

        # filter whitespace tokens out
        filtered_tokens, filtered_tags = zip(
            *[(t.text, tag) for t, tag in zip(tokenized_text, tags) if not t.text.isspace()]
        )

        row = deepcopy(row)
        row["text"] = filtered_tokens
        row["labels"] = filtered_tags
        processed_docs.append(row)

    return processed_docs


def doccano_to_tags(
    data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Converts doccano tags to a tokenized list of words with a list of labels per word."""
    nlp = ner_tokenizer()
    processed_docs: list[dict[str, Any]] = []
    for row in tqdm(data, desc="Converting offsets"):
        text = row["text"]
        labels = row["label"]

        tokenized_text = nlp(text)
        per_token_labels: dict[int, list[str]] = {}
        for start, end, lbl in labels:
            tags = offsets_to_biluo_tags(
                tokenized_text, [(start, end, lbl)]
            )

            if "-" in tags:
                print("Have misaligned tokens!")
                print("Misaligned text:", [t.text for t, tag in zip(tokenized_text, tags) if tag == "-"])
                print("Selected text:", [text[start:stop] for start, stop, _ in labels])
                print(f"ID: {row.get('id', 'unknown')}, UID: {row.get('uid', 'unknown')}")
                print(f"Report: {text}")

            # convert BILUO to BIO
            tags = biluo_to_bio_tags(tags, strict=True)

            # filter whitespace tokens out
            filtered_tokens, filtered_tags = zip(
                *[(t.text, tag) for t, tag in zip(tokenized_text, tags) if not t.text.isspace()]
            )

            # add label to per_token_labels
            for i, tag in enumerate(filtered_tags):
                if tag != "O":
                    per_token_labels[i] = per_token_labels.get(i, []) + [str(tag)]

        # filter whitespace tokens out
        text_parts: list[str] = [t.text for t in tokenized_text if not t.text.isspace()]

        # convert per_token_labels to a list of labels per token
        per_token_labels_flat = [
            [lbl for lbl in per_token_labels.get(i, ["O"])]
            for i in range(len(text_parts))
        ]

        row = deepcopy(row)
        row["text_parts"] = text_parts
        row["label"] = per_token_labels_flat
        processed_docs.append(row)

    return processed_docs


def doccano_tokenize(
    data: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Converts doccano data to a tokenized list of words."""
    nlp = ner_tokenizer()
    processed_docs = []
    for row in tqdm(data, desc="Tokenizing data"):
        text = row["text"]
        tokenized_text = nlp(text)

        # filter whitespace tokens out
        filtered_tokens = [t.text for t in tokenized_text if not t.text.isspace()]

        row = deepcopy(row)
        row["text_parts"] = filtered_tokens
        processed_docs.append(row)

    return processed_docs


def reconstruct_text(tokenized_text: list[spacy.tokens.Token]) -> str:
    reconstructed_text = ""

    # iterate through each token in the tokenized text
    for token in tokenized_text:
        # append the token's text
        reconstructed_text += token.text

        # Append any whitespaces and/or /n that followed this token
        if token.whitespace_:
            reconstructed_text += token.whitespace_

    return reconstructed_text


def fix_sequence_labels_after_anon(
    text_orig: str,
    text_anon: str,
    labels_orig: list[tuple[int, int, str]],
    expand_on_neighboring_insertions: bool = False,
    verbose: bool = False,
    strict: bool = True,
) -> pd.DataFrame:
    """Experimental function to fix sequence labels after anonymization. This function is not guaranteed to work."""
    from unidecode import unidecode
    labels_anon = deepcopy(labels_orig)

    if verbose:
        # show differences
        print("Text difference:")
        for text in difflib.unified_diff(text_orig.split(" "), text_anon.split(" ")):
            if text[:3] not in ('+++', '---', '@@ '):
                if text[:1] in ("-", "+"):
                    print(text)

        # show original labels
        for start, end, lbl in labels_orig:
            selected_text_orig = text_orig[start:end]
            print(f"{lbl:7} ({start:3d}, {end:3d}) {selected_text_orig}")

    # shift labels based on the differences between the original and anon text
    sequence_matcher = difflib.SequenceMatcher(None, text_orig, text_anon, autojunk=False)
    for i in range(len(labels_anon)):
        start, end, lbl = labels_anon[i]
        for tag, i1, i2, j1, j2 in sequence_matcher.get_opcodes():
            # Example: {tag} a[i1:i2] --> b[j1:j2]
            # delete    a[0:1] --> b[0:0]      'q' --> ''
            # equal     a[1:3] --> b[0:2]     'ab' --> 'ab'
            # replace   a[3:4] --> b[2:3]      'x' --> 'y'
            # equal     a[4:6] --> b[3:5]     'cd' --> 'cd'
            # insert    a[6:6] --> b[5:6]       '' --> 'f'
            if tag == "equal":
                continue
            elif tag == "insert":
                # shift anon labels to the right
                shift = j2 - j1
            elif tag == "delete":
                # shift anon labels to the left
                shift = i1 - i2
            elif tag == "replace":
                # shift anon labels
                shift = (j2 - j1) - (i2 - i1)
                if verbose:
                    print(f"Replace: {text_orig[i1:i2]} --> {text_anon[j1:j2]}")
            else:
                raise ValueError(f"Unexpected tag: {tag}")

            selected_text_anon = text_anon[j1:j2]
            selected_text_orig = text_orig[i1:i2]

            if end > j1 or (end == j1 and expand_on_neighboring_insertions):
                end = end + shift
            if start > j1 or (start == j1 and not expand_on_neighboring_insertions):
                start = start + shift

        # save shifted label
        labels_anon[i] = (start, end, lbl)

    if verbose:
        # show anon labels
        for start, end, lbl in labels_anon:
            selected_text_anon = text_anon[start:end]
            print(f"{lbl:7} ({start:3d}, {end:3d}) {selected_text_anon}")

    # verify labels
    for (i1, i2, lbl1), (j1, j2, lbl2) in zip(labels_orig, labels_anon):
        selected_text_orig = unidecode(text_orig[i1:i2]).replace("  ", " ")
        selected_text_anon = unidecode(text_anon[j1:j2]).replace("  ", " ")
        if lbl1 != lbl2:
            raise ValueError(f"Labels {lbl1} and {lbl2} do not match.")
        if selected_text_orig != selected_text_anon:
            if strict:
                raise ValueError(f"Selected text {selected_text_orig} does not match {selected_text_anon}")
            else:
                print(f"Selected text {selected_text_orig} does not match {selected_text_anon}")

    return labels_anon
