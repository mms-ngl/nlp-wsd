import os
import numpy as np
import torch
from typing import List, Dict

from model import Model
from stud.wsd import WSDModule
from transformers import AutoTokenizer, BertModel

no_hononym = "no_hononym"

def build_model(device: str) -> Model:

    model_folder = "../../model/"
    label2id_name = "label2id.pth"

    model_name = "model_weights.pth"

    label2id_path = os.path.join(os.path.dirname(__file__), model_folder + label2id_name)
    model_path = os.path.join(os.path.dirname(__file__), model_folder + model_name)

    return StudentModel(label2id_path, model_path, device)


class StudentModel(Model):

    def __init__(self, 
                 label2id_path:str, 
                 model_path:str, 
                 device:str="cpu"):

        self.device = device

        model_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.label2id = torch.load(label2id_path)
        print("len(self.label2id.keys()): ", len(self.label2id.keys()))

        self.wsd_model = WSDModule(
            model_name=model_name, 
            num_labels=len(self.label2id.keys()), 
            fine_tune_lm=False).to(self.device)

        self.wsd_model.load_state_dict(torch.load(model_path))

        # set the model in evaluation mode
        self.wsd_model.eval()

    def predict(self, sentences: List[Dict]) -> List[List[str]]:

        batch = self.tokenizer(
            [sentence["words"] for sentence in sentences],
            return_tensors="pt",
            padding=True,
            # We use this argument because the texts in our dataset are lists of words.
            is_split_into_words=True
        )

        # to guarantee the tensors are on the same device as the model
        batch = batch.to(self.device)
        print("batch: ", batch)


        with torch.no_grad():
            output = self.wsd_model(**batch)

        predictions = output["probabilities"]


        selects_batch = []

        for i, sentence in enumerate(sentences):

            prediction = predictions[i]

            word_ids = batch.word_ids(batch_index=i)

            instance_ids = sentence["instance_ids"]

            candidates = sentence["candidates"]

            selects = []

            for instance_id in instance_ids:

                instance_candidates = candidates[instance_id]

                input_id = word_ids.index(int(instance_id))
                
                labels_probabilities = []

                # getting softmax probabilities of candidate senses
                for candidate in instance_candidates:
                    if candidate in self.label2id:
                        print("candidate: ", candidate)
                        
                        instance_prediction = prediction[input_id][self.label2id[candidate]]

                        labels_probabilities.append(instance_prediction)

                    else:
                        labels_probabilities.append(0)

                # selecting label id that has max softmax probability
                select_idx = labels_probabilities.index(max(labels_probabilities))

                select = instance_candidates[select_idx]

                selects.append(select)

            selects_batch.append(selects)

        return selects_batch