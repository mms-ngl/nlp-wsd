import json
import logging
from rich.progress import track

logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

import argparse
import requests
import time

from requests.exceptions import ConnectionError
from typing import Tuple, List, Any, Dict


def get_n_instances(l: List[List[Any]]) -> int:
    return sum(len(inner_l) for inner_l in l)


def wsd_accuracy_score(senses_s: List[List[List[str]]], predictions_s: List[List[str]]) -> float:
    if len(senses_s) != len(predictions_s):
        raise ValueError(f"The number of input sents and the number of sents returned in predictions do not match: # "
                         f"input sents = {len(senses_s)}, # returned sents = {len(predictions_s)}")
    n_instances = get_n_instances(senses_s)
    print(f"# instances: {n_instances}")
    correct = 0
    for i, (senses, predictions) in enumerate(zip(senses_s, predictions_s)):
        if len(senses) != len(predictions):
            raise ValueError(
                f"For the sentence with idx {i}, the number of input WSD instances and the number of WSD instances "
                f"returned in predictions do not match: # input instances = {len(senses)}, # returned instances = "
                f"{len(predictions)}")
        for sense, prediction in zip(senses, predictions):
            if prediction in sense:
                correct += 1
    return correct / n_instances


def read_dataset(path: str) -> Tuple[List[Dict], List[List[List[str]]]]:
    sentences_s, senses_s = [], []

    with open(path) as f:
        data = json.load(f)

    for sentence_id, sentence_data in data.items():
        assert len(sentence_data["instance_ids"]) > 0
        assert (len(sentence_data["instance_ids"]) ==
                len(sentence_data["senses"]) ==
                len(sentence_data["candidates"]))
        assert all(len(gt) > 0 for gt in sentence_data["senses"].values())
        assert (all(gt_sense in candidates for gt_sense in gt)
                for gt, candidates in zip(sentence_data["senses"].values(), sentence_data["candidates"].values()))
        assert len(sentence_data["words"]) == len(sentence_data["lemmas"]) == len(sentence_data["pos_tags"])
        senses_s.append(list(sentence_data.pop("senses").values()))
        sentence_data["id"] = sentence_id
        sentences_s.append(sentence_data)

    assert len(sentences_s) == len(senses_s)

    return sentences_s, senses_s


def main(test_path: str, endpoint: str, batch_size=32):
    try:
        sentences_s, senses_s = read_dataset(test_path)
    except FileNotFoundError as e:
        logging.error(f"Evaluation crashed because {test_path} does not exist")
        exit(1)
    except Exception as e:
        logging.error(
            f"Evaluation crashed. Most likely, the file you gave is not in the correct format"
        )
        logging.error(f"Printing error found")
        logging.error(e, exc_info=True)
        exit(1)

    # max_try = 10
    max_try = 100
    iterator = iter(range(max_try))

    while True:

        try:
            i = next(iterator)
        except StopIteration:
            logging.error(
                f"Impossible to establish a connection to the server even after 10 tries"
            )
            logging.error(
                "The server is not booting and, most likely, you have some error in build_model or StudentClass"
            )
            logging.error(
                "You can find more information inside logs/. Checkout both server.stdout and, most importantly, "
                "server.stderr"
            )
            exit(1)

        logging.info(f"Waiting 10 second for server to go up: trial {i}/{max_try}")
        time.sleep(10)

        try:
            response = requests.post(
                endpoint, json={"sentences_s": [
                    {
                        "id": "d000.s032",
                        "instance_ids": {"0": "d000.s032.t000", "3": "d000.s032.t001"},
                        "lemmas": ["choose", "203", "business", "executive", "."],
                        "pos_tags": ["VERB", "NUM", "NOUN", "NOUN", "."],
                        "words": ["Choose", "203", "business", "executives", "."],
                        'candidates': {"0": ["select.v.h.01", "preferred.v.h.01", "chosen.v.h.01"],
                                       "3": ["executive.n.h.01"]}
                    }
                ]}
            ).json()
            response["predictions_s"]
            logging.info("Connection succeded")
            break
        except ConnectionError as e:
            continue
        except KeyError as e:
            logging.error(f"Server response in wrong format")
            logging.error(f"Response was: {response}")
            logging.error(e, exc_info=True)
            exit(1)

    predictions_s = []

    for i in track(range(0, len(sentences_s), batch_size), description="Evaluating"):
        batch = sentences_s[i: i + batch_size]
        try:
            response = requests.post(endpoint, json={"sentences_s": batch}).json()
            predictions_s += response["predictions_s"]
        except KeyError as e:
            logging.error(f"Server response in wrong format")
            logging.error(f"Response was: {response}")
            logging.error(e, exc_info=True)
            exit(1)

    acc = wsd_accuracy_score(senses_s, predictions_s)

    print(f"# accuracy: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file", type=str, help="File containing data you want to evaluate upon"
    )
    args = parser.parse_args()

    main(test_path=args.file, endpoint="http://127.0.0.1:12345")