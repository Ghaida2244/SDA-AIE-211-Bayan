"""Lab 1 starter: sentence segmentation."""

import spacy
from bayan.preprocessing.core import preprocess


def build_pipeline():
    nlp = spacy.blank("xx")
    nlp.add_pipe("sentencizer")

    return nlp


def split_sentences(raw: str, nlp) -> list[str]:
    text = preprocess(raw)
    doc = nlp(text)

    return [
      sent.text.strip()
       for sent in doc.sents
       if sent.text.strip()
    ]
