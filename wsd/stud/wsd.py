import os
import json
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim

from transformers import AutoTokenizer, BertModel

from typing import List, Dict
import matplotlib.pyplot as plt

homonyms_path = "../../data/map/coarse_fine_defs_map.json"

train_path = "../../data/coarse-grained/train_coarse_grained.json"
val_path = "../../data/coarse-grained/val_coarse_grained.json"

save_path = "../../model/"

# model_name = "distilbert-base-uncased"
model_name = "bert-base-uncased"

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device('cpu')
print("Using {}.".format(device))

no_homonym = "no_homonym"

batch_size = 32
training_number = 50

# Building labels' vocabulary.
def get_homonyms(path: str) -> List[Dict]:
    labels = []

    labels.append(no_homonym)

    with open(path) as f:
        data = json.load(f)

    for homonym, homonym_data in data.items():
        labels.append(homonym)

    return labels


def read_dataset(path: str) -> List[Dict]:
    dataset = []

    with open(path) as f:
        data = json.load(f)

    for sentence_id, sentence_data in data.items():
        sentence_data["id"] = sentence_id
        dataset.append(sentence_data)

    return dataset


# Defining our collate function.
def collate_fn(batch) -> Dict[str, torch.Tensor]:

    input_batch = tokenizer(
        [sentence["words"] for sentence in batch],
        return_tensors="pt",
        padding=True,
        # We use this argument because the texts in our dataset are lists of words.
        is_split_into_words=True,
    )

    labels = []
    senses = [sentence["senses"] for sentence in batch]

    for i, label in enumerate(senses):

        # Obtains the word_ids of the i-th sentence.
        word_ids = input_batch.word_ids(batch_index=i)

        previous_word_idx = None

        label_ids = []

        senses_ids = [int(i) for i in label.keys()]

        for word_idx in word_ids:

            # Special tokens have a word id that is None. We set the label to -100,
            # so they are automatically ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)

            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:

                if word_idx in senses_ids:
                    homonym = label[str(word_idx)][0]
                    label_ids.append(label2id[homonym])
                else:
                    label_ids.append(label2id[no_homonym])

            # For the other tokens in a word, we set the label to -100,
            # so they are automatically ignored in the loss function.
            else:
                label_ids.append(-100)

            previous_word_idx = word_idx

        labels.append(label_ids)

    input_batch["labels"] = torch.as_tensor(labels)

    return input_batch


# Model definition
class WSDModule(torch.nn.Module):
    def __init__(self, model_name: str, num_labels: int, fine_tune_lm: bool = True):
        super(WSDModule, self).__init__()

        self.transformer_model = BertModel.from_pretrained(model_name)

        if not fine_tune_lm:
            for param in self.transformer_model.parameters():
                param.requires_grad = False

        # self.dropout = torch.nn.Dropout(0.2)

        self.classifier = torch.nn.Linear(self.transformer_model.config.hidden_size, num_labels, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None):

        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }

        transformers_output = self.transformer_model(**model_kwargs)

        logits = self.classifier(transformers_output.last_hidden_state)

        # Softmax on top of Classifier layer.
        probabilities = torch.softmax(logits, dim=-1)

        output = {'logits': logits, 'probabilities': probabilities}

        return output


# Model training
class Trainer():

    def __init__(self,
                 model:nn.Module,
                 loss_function,
                 optimizer):
        """
        Args:
            model: the model we want to train.
            loss_function: the loss_function to minimize.
            optimizer: the optimizer used to minimize the loss_function.
        """

        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer

    def train_model(self, train_dataset:Dataset, val_dataset:Dataset, epochs:int=1):
        """
        Args:
            train_dataset: a Dataset or DatasetLoader instance containing
                the training instances.
            val_dataset: a Dataset or DatasetLoader instance used to evaluate
                learning progress.
            epochs: the number of times to iterate over train_dataset.
        """
        training_loss, validation_loss = [], []

        for epoch in range(epochs):
            print(' Epoch {:03d}'.format(epoch + 1))

            train_loss = self.train(train_dataset)
            print('\t[E: {:2d}] train loss = {:0.4f}'.format(epoch+1, train_loss))
            training_loss.append(train_loss)

            valid_loss = self.validate(val_dataset)
            print('  [E: {:2d}] valid loss = {:0.4f}'.format(epoch+1, valid_loss))
            validation_loss.append(valid_loss)

            # if epoch % 10 == 0 and epoch > 0:
            #     torch.save(self.model.state_dict(), save_path + str(epoch) + '_model_weights.pth')

        return training_loss, validation_loss

    def train(self, train_dataset):
        """
        Args:
            train_dataset: the dataset to use to train the model.

        Returns:
            the average train loss over train_dataset.
        """

        train_loss = 0.0
        self.model.train()

        for batch in train_dataset:

            batch.to(device)

            labels = batch.pop('labels')

            self.optimizer.zero_grad()

            """
            ** operator converts batch items in named arguments,
            (e.g. 'input_ids', 'attention_mask_ids' ...),
            taken as input by the model forward pass.
            """
            output = self.model(**batch)
            preds = output["logits"]
            preds = preds.view(-1, preds.shape[-1])

            labels = labels.view(-1)

            loss = self.loss_function(preds, labels)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.tolist()

        return train_loss / len(train_dataset)

    def validate(self, val_dataset):
        """
        Args:
            val_dataset: the dataset to use to evaluate the model

        Returns:
            the average validation loss over val_dataset
        """

        valid_loss = 0.0
        self.model.eval()

        with torch.no_grad():

            for batch in val_dataset:
                batch.to(device)

                labels = batch.pop('labels')

                output = self.model(**batch)
                preds = output["logits"]
                preds = preds.view(-1, preds.shape[-1])
                labels = labels.view(-1)

                loss = self.loss_function(preds, labels)
                valid_loss += loss.tolist()

        return valid_loss / len(val_dataset)


if __name__ == "__main__":

    # Initializations and functions calls.

    labels = get_homonyms(homonyms_path)
    label2id = {n: i for i, n in enumerate(labels)}

    train_dataset  = read_dataset(train_path)
    val_dataset  = read_dataset(val_path)

    # Instantiate the model
    wsd_model = WSDModule(model_name, len(label2id.keys()), fine_tune_lm=False)
    wsd_model.to(device)

    # Instantiate the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False
    )

    trainer = Trainer(
      model = wsd_model,
      loss_function = nn.CrossEntropyLoss(ignore_index=-100),
      optimizer = optim.Adam(wsd_model.parameters())
    )

    training_loss, validation_loss = trainer.train_model(train_dataloader, val_dataloader, training_number)

    # Savings final training results.
    torch.save(label2id, save_path + 'label2id.pth')
    torch.save(wsd_model.state_dict(), save_path + 'model_weights.pth')

    # Losses graphs
    plt.plot(range(1, training_number+1), training_loss, label='Train')
    plt.plot(range(1, training_number+1), validation_loss, label='Validation')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train and Validation Loss")
    plt.legend()
    plt.show()
