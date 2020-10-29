import torch
import torch.nn as nn
import torch.nn.functional as F


from transformers import BertForSequenceClassification

from Model.utils import replace_masked


class MyBertModel(nn.Module):

    def __init__(self, config):
        super(MyBertModel, self).__init__()
        self.config = config
        self.Roberta = BertForSequenceClassification.from_pretrained('RoBERTa_zh_Large_PyTorch/.')
        self.activation = nn.Tanh()
        self.pool = nn.Linear(self.config.bert_size, self.config.bert_size)

        self._classification = nn.Sequential(nn.Dropout(p=self.config.dropout),
                                             nn.Linear(self.config.bert_size,
                                             self.config.class_num))

        self.Roberta_config = self.Roberta.config

    def forward(self, p_h_tensor):
        input_id = p_h_tensor[0]
        segment = p_h_tensor[1]
        bpe = p_h_tensor[2]
        add_bpe = p_h_tensor[3]
        p_attention_mask = torch.zeros_like(input_id)
        for idx in range(p_attention_mask.size(0)):
            for jdx, value in enumerate(input_id[idx]):
                if value > 0:
                    p_attention_mask[idx][jdx] = 1
        _, p_hidden_states = self.Roberta(input_id, attention_mask=p_attention_mask, token_type_ids=segment)
        last_hidden = p_hidden_states[-1]

        first_token_tensor = last_hidden[:, 0]
        pooled_output = self.pool(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        #pooled_output = F.avg_pool1d(class_hidden, class_hidden.size(2)).squeeze(2)
        logits = self._classification(pooled_output)
        return logits
