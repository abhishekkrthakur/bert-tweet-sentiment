import config
import dataset
import engine
import torch
import utils
import params
import sandesh
import pandas as pd
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler

from model import TweetModel
from sklearn import model_selection
from sklearn import metrics
import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from apex import amp


def run():
    dfx = pd.read_csv(config.TRAINING_FILE)
    dfx = dfx.dropna().reset_index(drop=True)

    df_train, df_valid = model_selection.train_test_split(
        dfx,
        test_size=0.20,
        random_state=42,
        stratify=dfx.sentiment.values
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = dataset.TweetDataset(
        tweet=df_train.text.values,
        sentiment=df_train.sentiment.values,
        selected_text=df_train.selected_text.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )

    valid_dataset = dataset.TweetDataset(
        tweet=df_valid.text.values,
        sentiment=df_valid.sentiment.values,
        selected_text=df_valid.selected_text.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=0
    )

    device = torch.device("cuda")
    model_config = transformers.BertConfig.from_pretrained(config.BERT_PATH)
    model_config.output_hidden_states = True
    model = TweetModel(conf=model_config)
    model.to(device)

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    
    optimizer = AdamW(params.optimizer_params(model), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    model = nn.DataParallel(model)

    es = utils.EarlyStopping(patience=5, mode="max")
    sandesh.send("Training is Starting")
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_data_loader, model, optimizer, device, scheduler=scheduler)
        jaccard = engine.eval_fn(valid_data_loader, model, device)
        print(f"Jaccard Score = {jaccard}")
        sandesh.send(f"Epoch={epoch}, Jaccard={jaccard}")
        es(jaccard, model, model_path="model.bin")
        if es.early_stop:
            print("Early stopping")
            break


if __name__ == "__main__":
    run()