import config
import torch
import transformers
import torch.nn as nn
from torch.nn import functional as F


class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, conf):
        super(TweetModel, self).__init__(conf)
        self.bert = transformers.BertModel(conf)
        self.drop_out = nn.Dropout(0.3)
        self.l0 = nn.Linear(768 * 4, 2)

        self.init_weights()
    
    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.bert(
            ids, 
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        out = torch.cat((out[-1], out[-2], out[-3], out[-4]), dim=-1)
        out = self.drop_out(out)
        logits = self.l0(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
