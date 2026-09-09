"""Lab 4 starter: per-model Arabic normalisation profiles."""
from dataclasses import dataclass

import re
import unicodedata
from dataclasses import dataclass

from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tokenizers.morphological import MorphologicalTokenizer


# Match Arabic diacritics and Quranic annotation marks
_ARABIC_DIACRITICS_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)

# Match repeated whitespace
_WHITESPACE_RE = re.compile(r"\s+")

# Load the morphological tokenizer only when segmentation is requested
_MORPH_TOKENIZER = None


@dataclass(frozen=True)
class ArabicProfile:
    """Define the Arabic normalization rules used by a model."""

    name: str
    dediacritize: bool = False


def normalize_arabic(
    text: str,
    profile: ArabicProfile,
) -> str:
    """Normalize Arabic text according to the selected profile."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    # Preserve a clean display version with minimal normalization
    text = unicodedata.normalize("NFC", text)

    if profile.name == "display":
        return _WHITESPACE_RE.sub(
            " ",
            text,
        ).strip()

    if profile.name != "bayan_ar_v1":
        raise ValueError(
            f"Unsupported Arabic profile: {profile.name}"
        )

    # Remove Arabic tatweel
    text = text.replace("ـ", "")

    # Remove Arabic diacritics when requested by the profile
    if profile.dediacritize:
        text = _ARABIC_DIACRITICS_RE.sub(
            "",
            text,
        )

    # Normalize common Arabic letter variants
    text = re.sub(
        r"[أإآٱ]",
        "ا",
        text,
    )

    text = text.replace("ؤ", "و")
    text = text.replace("ئ", "ي")
    text = text.replace("ى", "ي")
    text = text.replace("ة", "ه")

    # Normalize whitespace
    text = _WHITESPACE_RE.sub(
        " ",
        text,
    ).strip()

    return text

def segment(text: str) -> list[str]:
    """Split Arabic words into clitics using CAMeL Tools D3 tokenization."""

    global _MORPH_TOKENIZER

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    words = text.split()

    if not words:
        return []

    # Reuse the tokenizer because loading the morphology model is expensive
    if _MORPH_TOKENIZER is None:
        disambiguator = MLEDisambiguator.pretrained()

        _MORPH_TOKENIZER = MorphologicalTokenizer(
            disambiguator,
            scheme="d3tok",
            split=True,
            diac=False,
        )

    return _MORPH_TOKENIZER.tokenize(words)