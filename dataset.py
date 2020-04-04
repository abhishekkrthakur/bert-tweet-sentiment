import config
import torch
import pandas as pd
import numpy as np


def process_data(tweet, selected_text, sentiment, tokenizer, max_len):    
    tweet = " ".join(str(tweet).split())
    selected_text = " ".join(str(selected_text).split())

    len_st = len(selected_text)
    idx0 = None
    idx1 = None
    for ind in (i for i, e in enumerate(tweet) if e == selected_text[0]):
        if tweet[ind: ind+len_st] == selected_text:
            idx0 = ind
            idx1 = ind + len_st
            break

    char_targets = [0] * len(tweet)
    if idx0 != None and idx1 != None:
        for ct in range(idx0, idx1):
            char_targets[ct] = 1
    
    tok_tweet = tokenizer.encode(tweet)
    input_ids_orig = tok_tweet.ids[1:-1]
    tweet_offsets = tok_tweet.offsets[1:-1]
    
    target_idx = []
    for j, (offset1, offset2) in enumerate(tweet_offsets):
        if sum(char_targets[offset1: offset2]) > 0:
            target_idx.append(j)
    
    targets_start = target_idx[0]
    targets_end = target_idx[-1]

    sentiment_id = {
        'positive': 3893,
        'negative': 4997,
        'neutral': 8699
    }
    
    input_ids = [101] + [sentiment_id[sentiment]] + [102] + input_ids_orig + [102]
    token_type_ids = [0, 0, 0] + [1] * (len(input_ids_orig) + 1)
    mask = [1] * len(token_type_ids)
    tweet_offsets = [(0, 0)] * 3 + tweet_offsets + [(0, 0)]
    targets_start += 3
    targets_end += 3

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        tweet_offsets = tweet_offsets + ([(0, 0)] * padding_length)
    
    return {
        'ids': input_ids,
        'mask': mask,
        'token_type_ids': token_type_ids,
        'targets_start': targets_start,
        'targets_end': targets_end,
        'orig_tweet': tweet,
        'orig_selected': selected_text,
        'sentiment': sentiment,
        'offsets': tweet_offsets
    }


class TweetDataset:
    def __init__(self, tweet, sentiment, selected_text):
        self.tweet = tweet
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN
    
    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        data = process_data(
            self.tweet[item], 
            self.selected_text[item], 
            self.sentiment[item],
            self.tokenizer,
            self.max_len
        )

        return {
            'ids': torch.tensor(data["ids"], dtype=torch.long),
            'mask': torch.tensor(data["mask"], dtype=torch.long),
            'token_type_ids': torch.tensor(data["token_type_ids"], dtype=torch.long),
            'targets_start': torch.tensor(data["targets_start"], dtype=torch.long),
            'targets_end': torch.tensor(data["targets_end"], dtype=torch.long),
            'orig_tweet': data["orig_tweet"],
            'orig_selected': data["orig_selected"],
            'sentiment': data["sentiment"],
            'offsets_start': torch.tensor([x for x, _ in data["offsets"]], dtype=torch.long),
            'offsets_end': torch.tensor([x for _, x in data["offsets"]], dtype=torch.long)
        }


if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE)
    df = df.dropna().reset_index(drop=True)
    dset = TweetDataset(tweet=df.text.values, sentiment=df.sentiment.values, selected_text=df.selected_text.values)
    print(dset[100])
    #for j in range(len(dset)):
    #    print(j)
    #    print(dset[j])