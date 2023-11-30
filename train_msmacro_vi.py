import sys
import json
from sentence_transformers import SentenceTransformer, LoggingHandler, models, losses, InputExample
import logging
from datetime import datetime
import os
import argparse
import math
import unicodedata
from torch.utils.data import Dataset, DataLoader, Subset
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import models, losses
from torch.utils.data import DataLoader


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

parser = argparse.ArgumentParser()
parser.add_argument("--train_batch_size", default=8, type=int)
parser.add_argument("--max_seq_length", default=32, type=int)
parser.add_argument("--model_name", required=True)
parser.add_argument("--max_passages", default=0, type=int)
parser.add_argument("--epochs", default=1, type=int)
parser.add_argument("--pooling", default="mean")
parser.add_argument("--size_dataset", default=1, type=float)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--scheduler", default='WarmupLinear', type=str)
parser.add_argument("--weight_decay", default=0.1, type=float)
args = parser.parse_args()

logging.info(str(args))


train_batch_size = args.train_batch_size         
model_name = args.model_name
max_passages = args.max_passages
max_seq_length = args.max_seq_length            
num_epochs = args.epochs
size_dataset = args.size_dataset


model_save_path = f'output/train_msmacro_vi'

train_vi = {}  
json_train_vi_filepath = "/kaggle/input/qna-vietnamese/QnA/data_mrc/msmarco/vi_msmarco_train.json"

if not os.path.exists(json_train_vi_filepath):
    logging.error(f"JSON file not found: {json_train_vi_filepath}")

logging.info(f"Read train data from JSON file: {json_train_vi_filepath}")
with open(json_train_vi_filepath, 'r', encoding='utf-8') as json_file:
    train_vi = json.load(json_file)

def preprocess_text(text):
    text = text.replace("_", " ")
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    text = ' '.join(text.split())
    return text

def preprocess_item(item):
    item["question"] = preprocess_text(item["question"])
    item["context"] = preprocess_text(item["context"])
    return item

preprocessed_data = [preprocess_item(item) for item in train_vi]

subset_size = int(size_dataset * len(preprocessed_data)) 
subset_train_dataset = Subset(preprocessed_data, range(subset_size))

word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


train_data = [
    InputExample(texts=[item["question"], item["context"]]) if item["label"] == 0
    else InputExample(texts=[item["question"], item["context"]],
                      label=item["label"])
    for item in subset_train_dataset
]

train_dataloader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
train_loss = losses.MultipleNegativesRankingLoss(model)
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)

model.fit(train_objectives=[(train_dataloader, train_loss)],
          epochs=num_epochs,
          scheduler=args.scheduler,
          warmup_steps=warmup_steps,
          checkpoint_path=model_save_path,
          use_amp=True,
          optimizer_params = {'lr': args.lr},
          weight_decay=args.weight_decay,
          save_best_model= True,
          checkpoint_save_total_limit = 1
          )

model.save(model_save_path)

