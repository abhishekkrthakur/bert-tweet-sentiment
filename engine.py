import utils
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import re
import config

from apex import amp


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    loss_fct = nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss) / 2
    return total_loss


def train_fn(data_loader, model, optimizer, device, scheduler=None):
    model.train()
    model.zero_grad()
    losses = utils.AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))
    
    for bi, d in enumerate(tk0):

        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets_start = d["targets_start"]
        targets_end = d["targets_end"]
        sentiment = d["sentiment"]

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets_start = targets_start.to(device, dtype=torch.long)
        targets_end = targets_end.to(device, dtype=torch.long)

        outputs_start, outputs_end = model(
            ids=ids,
            mask=mask,
            token_type_ids=token_type_ids,
        )
        loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), 1.0)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        model.zero_grad()

        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg)

def generate_eval_preds(data_loader, model, device):
    model.eval()
    losses = utils.AverageMeter()
    fin_outputs_start = []
    fin_outputs_end = []
    fin_orig_selected = []
    fin_orig_sentiment = []
    fin_orig_tweet = []
    fin_offsets_start = []
    fin_offsets_end = []
    
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            sentiment = d["sentiment"]
            orig_selected = d["orig_selected"]
            orig_tweet = d["orig_tweet"]
            targets_start = d["targets_start"]
            targets_end = d["targets_end"]
            offsets_start = d["offsets_start"]
            offsets_end = d["offsets_end"]

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets_start = targets_start.to(device, dtype=torch.long)
            targets_end = targets_end.to(device, dtype=torch.long)

            outputs_start, outputs_end = model(
                ids=ids,
                mask=mask,
                token_type_ids=token_type_ids
            )
            loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)
            
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg)
            
            fin_outputs_start.append(torch.softmax(outputs_start, dim=1).cpu().detach().numpy())
            fin_outputs_end.append(torch.softmax(outputs_end, dim=1).cpu().detach().numpy())
            fin_offsets_start.append(offsets_start.numpy())
            fin_offsets_end.append(offsets_end.numpy())

            fin_orig_sentiment.extend(sentiment)
            fin_orig_selected.extend(orig_selected)
            fin_orig_tweet.extend(orig_tweet)

    fin_outputs_start = np.vstack(fin_outputs_start)
    fin_outputs_end = np.vstack(fin_outputs_end)
    fin_offsets_start = np.vstack(fin_offsets_start)
    fin_offsets_end = np.vstack(fin_offsets_end)

    return (fin_outputs_start, fin_outputs_end, fin_orig_selected, fin_orig_sentiment,
            fin_orig_tweet, fin_offsets_start, fin_offsets_end)


def eval_fn(data_loader, model, device):
    (fin_outputs_start, 
    fin_outputs_end, 
    fin_orig_selected, 
    fin_orig_sentiment, 
    fin_orig_tweet, 
    fin_offsets_start,
    fin_offsets_end) = generate_eval_preds(data_loader, model, device)
    jaccards = []

    for j in range(fin_outputs_start.shape[0]):
        target_string = fin_orig_selected[j]
        sentiment_val = fin_orig_sentiment[j]
        original_tweet = fin_orig_tweet[j]
        offsets_start = fin_offsets_start[j, :].tolist()
        offsets_end = fin_offsets_end[j, :].tolist()
        offsets = list(zip(offsets_start, offsets_end))

        idx_start = np.argmax(fin_outputs_start[j, :])
        idx_end = np.argmax(fin_outputs_end[j, :])
        
        if idx_end < idx_start:
            idx_end = idx_start
        
        filtered_output  = ""
        original_tweet = " ".join(original_tweet.split())
        for ix in range(idx_start, idx_end + 1):
            if offsets[ix][0] == 0 and offsets[ix][1] == 0:
                continue
            filtered_output += original_tweet[offsets[ix][0]: offsets[ix][1]]
            if (ix+1) < len(offsets) and offsets[ix][1] < offsets[ix+1][0]:
                filtered_output += " "

        if sentiment_val == "neutral":
            filtered_output = original_tweet

        jac = utils.jaccard(target_string.strip(), filtered_output.strip())
        jaccards.append(jac)

    mean_jaccard = np.mean(jaccards)
    return mean_jaccard
