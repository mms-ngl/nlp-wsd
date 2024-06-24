import os
import numpy as np
import torch
from typing import List, Dict

from model import Model
from stud.wsd import WSDModule
from transformers import AutoTokenizer

no_homonym = "no_homonym"

def build_model(device: str) -> Model:

    model_folder = "../../model/"

    label2id_name = "label2id.pth"

    model_name = "model_weights.pth"

    label2id_path = os.path.join(os.path.dirname(__file__), model_folder + label2id_name)
    model_path = os.path.join(os.path.dirname(__file__), model_folder + model_name)

    return StudentModel(label2id_path, model_path, device)


class StudentModel(Model):

    def __init__(self, 
                 label2id_path: str, 
                 model_path: str, 
                 device: str="cpu"):

        self.device = torch.device(device)

        model_name = "bert-base-uncased"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.label2id = torch.load(label2id_path, map_location=self.device)

        self.wsd_model = WSDModule(
            model_name=model_name, 
            num_labels=len(self.label2id.keys()), 
            fine_tune_lm=False).to(self.device)

        self.wsd_model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Set the model in evaluation mode.
        self.wsd_model.eval()

    def predict(self, sentences: List[Dict]) -> List[List[str]]:

        batch = self.tokenizer(
            [sentence["words"] for sentence in sentences],
            return_tensors="pt",
            padding=True,
            # We use this argument because the texts in our dataset are lists of words.
            is_split_into_words=True
        )

        # To guarantee the tensors are on the same device as the model.
        batch = batch.to(self.device)


        with torch.no_grad():
            output = self.wsd_model(**batch)

        predictions = output["probabilities"]

        # Batch of selections among candidate homonyms.
        selects_batch = []
        for i, sentence in enumerate(sentences):

            prediction = predictions[i]

            word_ids = batch.word_ids(batch_index=i)

            instance_ids = sentence["instance_ids"]

            candidates = sentence["candidates"]

            selects = []
            for instance_id in instance_ids:

                # Getting candidates of specific instance by id number.
                instance_candidates = candidates[instance_id]

                input_id = word_ids.index(int(instance_id))
                
                labels_probabilities = []

                # Getting softmax probabilities values of candidate homonyms.
                for candidate in instance_candidates:

                    # If candidate homonym from test dataset is in our labels' vocabulary.
                    if candidate in self.label2id:
                        
                        instance_prediction = prediction[input_id][self.label2id[candidate]]

                        labels_probabilities.append(instance_prediction)

                    # If candidate homonym from test dataset isn't in our labels' vocabulary,
                    # then we append value 0. 
                    else:
                        labels_probabilities.append(0)

                # Selecting label id that has max softmax probability.
                select_idx = labels_probabilities.index(max(labels_probabilities))

                select = instance_candidates[select_idx]

                selects.append(select)

            selects_batch.append(selects)

        return selects_batch