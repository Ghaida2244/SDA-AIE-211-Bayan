"""Lab 3 starter: NER label alignment."""


def align_labels(word_ids, word_labels):
    """Align word-level labels with the first subword of each token."""

    aligned_labels = []
    previous_word_id = None

    for word_id in word_ids:
        # Ignore special tokens such as CLS, SEP, and padding
        if word_id is None:
            aligned_labels.append(-100)

        # Assign the original label only to the first subword
        elif word_id != previous_word_id:
            aligned_labels.append(
                word_labels[word_id]
            )

        # Ignore any additional subwords from the same original word
        else:
            aligned_labels.append(-100)

        previous_word_id = word_id

    return aligned_labels