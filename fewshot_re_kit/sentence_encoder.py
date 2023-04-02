import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertModel


class BERTSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, cat_entity_rep=False, mask_entity=False):
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)  
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.cat_entity_rep = cat_entity_rep
        self.mask_entity = mask_entity
        self.tanh = nn.Tanh()
        self.linear1 = nn.Linear(768, 384)
        self.linear2 = nn.Linear(768, 384)

    def forward(self, inputs): 
        x = self.bert(inputs['word'], attention_mask=inputs['mask'])
        sqe_out = x[0]
        cls_out = x[1]

        for idx, (i, j) in enumerate(zip(inputs['pos1'], inputs['pos1_end'])):
            if idx == 0:
                ent_s = torch.mean(sqe_out[idx, i+1:j, :], dim=0, keepdim=True)
                ent_s = self.tanh(self.linear1(ent_s))
            else:
                ent_s_k_th = torch.mean(sqe_out[idx, i+1:j, :], dim=0, keepdim=True)
                ent_s_k_th = self.tanh(self.linear1(ent_s_k_th))
                ent_s = torch.cat((ent_s, ent_s_k_th), 0)
        for idxe, (h, t) in enumerate(zip(inputs['pos2'], inputs['pos2_end'])):
            if idxe == 0:
                ent_e = torch.mean(sqe_out[idxe, h+1:t, :], dim=0, keepdim=True)
                ent_e = self.tanh(self.linear2(ent_e))
            else:
                ent_e_k_th = torch.mean(sqe_out[idxe, h+1:t, :], dim=0, keepdim=True)
                ent_e_k_th = self.tanh(self.linear2(ent_e_k_th))
                ent_e = torch.cat((ent_e, ent_e_k_th), 0)

        outputs = self.bert(inputs['word'], attention_mask=inputs['mask'])
        tensor_range = torch.arange(inputs['word'].size()[0])  
        h_state = outputs[0][tensor_range, inputs["pos1"]]  
        t_state = outputs[0][tensor_range, inputs["pos2"]]
        state = torch.cat((h_state, t_state), -1)

        return sqe_out, ent_s, ent_e, cls_out


    def tokenize(self, raw_tokens, pos_head, pos_tail):
        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 1
        pos2_in_index = 1
        entity1_tail = 2
        entity2_tail = 2
        for token in raw_tokens:  
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens) 
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)  
            if self.mask_entity and ((pos_head[0] <= cur_pos and cur_pos <= pos_head[-1]) or (
                    pos_tail[0] <= cur_pos and cur_pos <= pos_tail[-1])):
                tokens += ['[unused4]']
            else:
                tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
                entity1_tail = len(tokens)
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
                entity2_tail = len(tokens)
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)  

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)  
        indexed_tokens = indexed_tokens[:self.max_length]  

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1  

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)
        entity1_tail = min(self.max_length, entity1_tail)
        entity2_tail = min(self.max_length, entity2_tail)

        return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask, entity1_tail-1, entity2_tail-1

    def tokenize_rel(self, raw_tokens):
        # token -> index
        tokens = ['[CLS]']
        name, description = raw_tokens
        for token in name:
            tokens += self.tokenizer.tokenize(token)
        tokens.append('[SEP]')
        for token in description:
            tokens += self.tokenizer.tokenize(token)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        mask[:len(tokens)] = 1

        return indexed_tokens, mask

class RoBERTaSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path, max_length, cat_entity_rep=False, mask_entity=False):
        nn.Module.__init__(self)
        self.roberta = BertModel.from_pretrained(pretrain_path)  
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(pretrain_path)
        self.cat_entity_rep = cat_entity_rep
        self.mask_entity = mask_entity
        self.tanh = nn.Tanh()
        self.linear1 = nn.Linear(768, 384)
        self.linear2 = nn.Linear(768, 384)

    def forward(self, inputs):  
        x = self.roberta(inputs['word'], attention_mask=inputs['mask'])
        sqe_out = x[0]
        cls_out = x[1]

        for idx, (i, j) in enumerate(zip(inputs['pos1'], inputs['pos1_end'])):
            if idx == 0:
                ent_s = torch.mean(sqe_out[idx, i+1:j, :], dim=0, keepdim=True)
                ent_s = self.linear1(self.tanh(ent_s))
            else:
                ent_s_k_th = torch.mean(sqe_out[idx, i+1:j, :], dim=0, keepdim=True)
                ent_s_k_th = self.linear1(self.tanh(ent_s_k_th))
                ent_s = torch.cat((ent_s, ent_s_k_th), 0)
        for idxe, (h, t) in enumerate(zip(inputs['pos2'], inputs['pos2_end'])):
            if idxe == 0:
                ent_e = torch.mean(sqe_out[idxe, h+1:t, :], dim=0, keepdim=True)
                ent_e = self.linear2(self.tanh(ent_e))
            else:
                ent_e_k_th = torch.mean(sqe_out[idxe, h+1:t, :], dim=0, keepdim=True)
                ent_e_k_th = self.linear2(self.tanh(ent_e_k_th))
                ent_e = torch.cat((ent_e, ent_e_k_th), 0)


        return sqe_out, ent_s, ent_e, cls_out


    def tokenize(self, raw_tokens, pos_head, pos_tail):
        tokens = ['[CLS]']
        cur_pos = 0
        pos1_in_index = 1
        pos2_in_index = 1
        entity1_tail = 2
        entity2_tail = 2
        for token in raw_tokens:  
            if cur_pos == pos_head[0]:
                tokens.append('[unused0]')
                pos1_in_index = len(tokens)  
            if cur_pos == pos_tail[0]:
                tokens.append('[unused1]')
                pos2_in_index = len(tokens)  
            if self.mask_entity and ((pos_head[0] <= cur_pos and cur_pos <= pos_head[-1]) or (
                    pos_tail[0] <= cur_pos and cur_pos <= pos_tail[-1])):
                tokens += ['[unused4]']
            else:
                tokens += self.tokenizer.tokenize(token)
            if cur_pos == pos_head[-1]:
                tokens.append('[unused2]')
                entity1_tail = len(tokens)
            if cur_pos == pos_tail[-1]:
                tokens.append('[unused3]')
                entity2_tail = len(tokens)
            cur_pos += 1
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)  

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)  
        indexed_tokens = indexed_tokens[:self.max_length]  

        # pos
        pos1 = np.zeros((self.max_length), dtype=np.int32)
        pos2 = np.zeros((self.max_length), dtype=np.int32)
        for i in range(self.max_length):
            pos1[i] = i - pos1_in_index + self.max_length
            pos2[i] = i - pos2_in_index + self.max_length

        # mask
        mask = np.zeros((self.max_length), dtype=np.int32)
        mask[:len(tokens)] = 1  

        pos1_in_index = min(self.max_length, pos1_in_index)
        pos2_in_index = min(self.max_length, pos2_in_index)
        entity1_tail = min(self.max_length, entity1_tail)
        entity2_tail = min(self.max_length, entity2_tail)

        return indexed_tokens, pos1_in_index - 1, pos2_in_index - 1, mask, entity1_tail-1, entity2_tail-1

    def tokenize_rel(self, raw_tokens):
        # token -> index
        tokens = ['[CLS]']
        name, description = raw_tokens
        for token in name:
            tokens += self.tokenizer.tokenize(token)
        tokens.append('[SEP]')
        for token in description:
            tokens += self.tokenizer.tokenize(token)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)

        # padding
        while len(indexed_tokens) < self.max_length:
            indexed_tokens.append(0)
        indexed_tokens = indexed_tokens[:self.max_length]

        # mask
        mask = np.zeros(self.max_length, dtype=np.int32)
        mask[:len(tokens)] = 1

        return indexed_tokens, mask
