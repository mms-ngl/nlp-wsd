from pprint import pp
from typing import List, Dict

from stud.implementation import build_model


def main(sentences: List[Dict]):

    model = build_model("cpu")
    predicted_sentences = model.predict(sentences)

    for sentence, predicted_labels_wsd in zip(sentences, predicted_sentences):
        print("Sentence dict:")
        pp(sentence)
        print()
        print("Predicted senses:")
        pp(predicted_labels_wsd)
        print()


if __name__ == "__main__":
    main([
            {
                "id": "d000.s032",
                "instance_ids": {"0": "d000.s032.t000", "3": "d000.s032.t001"},
                "lemmas": ["choose", "203", "business", "executive", "."],
                "pos_tags": ["VERB", "NUM", "NOUN", "NOUN", "."],
                "words": ["Choose", "203", "business", "executives", "."],
                'candidates': {"0": ["select.v.h.01", "preferred.v.h.01", "chosen.v.h.01"],
                               "3": ["executive.n.h.01"]}
            }
        ]
    )