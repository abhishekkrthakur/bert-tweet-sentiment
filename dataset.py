import config
import torch
import pandas as pd
import numpy as np


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
        tweet = " ".join(str(self.tweet[item]).split())
        selected_text = " ".join(str(self.selected_text[item]).split())
    
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
        
        tok_tweet = self.tokenizer.encode(tweet)
        tok_tweet_tokens = tok_tweet.tokens[1:-1]
        tok_tweet_ids_orig = tok_tweet.ids[1:-1]
        tok_tweet_offsets = tok_tweet.offsets[1:-1]
        
        target_idx = []
        for j, (offset1, offset2) in enumerate(tok_tweet_offsets):
            if sum(char_targets[offset1: offset2]) > 0:
                target_idx.append(j)
        
        targets_start = target_idx[0]
        targets_end = target_idx[-1]

        sentiment_id = {
            'positive': 3893,
            'negative': 4997,
            'neutral': 8699
        }

        tok_tweet_ids = [101] + [sentiment_id[self.sentiment[item]]] + [102] + tok_tweet_ids_orig + [102]
        token_type_ids = [0, 0, 0] + [1] * (len(tok_tweet_ids_orig) + 1)
        mask = [1] * len(token_type_ids)
        tok_tweet_offsets = [(0, 0)] * 3 + tok_tweet_offsets + [(0, 0)]
        targets_start += 3
        targets_end += 3

        padding_length = self.max_len - len(tok_tweet_ids)

        tok_tweet_ids = tok_tweet_ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        tok_tweet_offsets = tok_tweet_offsets + ([(0, 0)] * padding_length)

        return {
            'ids': torch.tensor(tok_tweet_ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets_start': torch.tensor(targets_start, dtype=torch.long),
            'targets_end': torch.tensor(targets_end, dtype=torch.long),
            'orig_tweet': self.tweet[item],
            'orig_selected': self.selected_text[item],
            'sentiment': self.sentiment[item],
            'offsets_start': torch.tensor([x for x, _ in tok_tweet_offsets], dtype=torch.long),
            'offsets_end': torch.tensor([x for _, x in tok_tweet_offsets], dtype=torch.long)
        }


if __name__ == "__main__":
    df = pd.read_csv(config.TRAINING_FILE)
    df = df.dropna().reset_index(drop=True)
    dset = TweetDataset(tweet=df.text.values, sentiment=df.sentiment.values, selected_text=df.selected_text.values)
    #print(dset[100])
    for j in range(len(dset)):
        print(j)
        print(dset[j])