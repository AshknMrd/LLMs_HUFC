from collections import OrderedDict
import os
import random
import warnings

import flwr as fl
import torch

from torch.utils.data import DataLoader

from datasets import load_dataset
from evaluate import load as load_metric

from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import logging

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.set_verbosity(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter('ignore')

DEVICE = torch.device("cpu")
print(f"The selected device is: {DEVICE}")
CHECKPOINT = "albert-base-v2"  # transformer model checkpoint
NUM_CLIENTS = 2
NUM_ROUNDS = 3
NUM_EPOCHS = 1
NUM_DATA_PER_CLIENT = 20 

def load_data(num_clients: int, num_samples_per_client: int = None):
    """Load IMDB data (training and eval)"""
    raw_datasets = load_dataset("imdb")
    raw_datasets = raw_datasets.shuffle(seed=42)

    # remove unnecessary data split
    del raw_datasets["unsupervised"]

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True)

    def split_indices(dataset_len):
        indices = random.sample(range(dataset_len), dataset_len) 
        if num_samples_per_client:
            total = num_clients * num_samples_per_client
            indices = indices[:total]
            return [indices[i * num_samples_per_client:(i + 1) * num_samples_per_client] for i in range(num_clients)]
        else:
            return [indices[i::num_clients] for i in range(num_clients)]  

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    train_indices = split_indices(len(tokenized_datasets["train"]))
    test_indices = split_indices(len(tokenized_datasets["test"]))

    tokenized_datasets = tokenized_datasets.remove_columns("text")
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainloader = [DataLoader(tokenized_datasets["train"].select(idxs), shuffle=True, batch_size=32, collate_fn=data_collator) for idxs in train_indices]
    testloader = [DataLoader(tokenized_datasets["test"].select(idxs), shuffle=False, batch_size=32, collate_fn=data_collator) for idxs in test_indices]

    return trainloader, testloader

def train(net, trainloader, epochs):
    optimizer = torch.optim.AdamW(net.parameters(), lr=5e-5)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = net(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def test(net, testloader):
    metric = load_metric("accuracy")
    loss = 0
    net.eval()
    for batch in testloader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.no_grad():
            outputs = net(**batch)
        logits = outputs.logits
        loss += outputs.loss.item()
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    loss /= len(testloader.dataset)
    accuracy = metric.compute()["accuracy"]
    return loss, accuracy

def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples), "loss": sum(losses) / sum(examples)}

class IMDBClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, testloader, epochs):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.epochs = epochs

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        print("Training Started...")
        train(self.net, self.trainloader, self.epochs)
        print("Training Finished.")
        return self.get_parameters(config={}), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return float(loss), len(self.testloader), {"accuracy": float(accuracy), "loss": float(loss)}

trainloader, testloader = load_data(num_clients=NUM_CLIENTS,num_samples_per_client=NUM_DATA_PER_CLIENT)
def client_fn(cid):
    cid = int(cid)
    print(f"\n \n client-{cid}\n \n")
    net = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT, num_labels=2).to(DEVICE)
    return IMDBClient(net, trainloader[cid], testloader[cid], NUM_EPOCHS)

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    evaluate_metrics_aggregation_fn=weighted_average,
)

fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
    client_resources={"num_cpus": 1, "num_gpus": 0},
    ray_init_args={"log_to_driver": False, "num_cpus": 1, "num_gpus": 0}
)