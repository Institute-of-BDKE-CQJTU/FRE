import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json


class FewRelDataset(data.Dataset):
    """
    FewRel Dataset
    """

    def __init__(self, name, encoder, N, K, Q, na_rate, root):

        self.root = root
        path = os.path.join(root, name + ".json")
        # print(path)
        if not os.path.exists(path):
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys()) 
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder

    def __getraw__(self, item):
        word, pos1, pos2, mask, pos1_end, pos2_end = self.encoder.tokenize(item['tokens'],
                                                       item['h'][1],
                                                       item['t'][1])
        return word, pos1, pos2, mask, pos1_end, pos2_end  
    def __getrel__(self, item):
        word, mask = self.encoder.tokenize_rel(item)
        return word, mask


    def __additem__(self, d, word, pos1, pos2, mask, pos1_end, pos2_end):
        d['word'].append(word)  
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)
        d['pos1_end'].append(pos1_end)
        d['pos2_end'].append(pos2_end)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N) 
        support_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'pos1_end': [], 'pos2_end': []}
        query_set = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'pos1_end': [], 'pos2_end': []}
        id_set = {'class_name': [], 'instance': []}
        relation_set = {'word': [], 'mask': []}
        query_label = []
        Q_na = int(self.na_rate * self.Q)
        na_classes = list(filter(lambda x: x not in target_classes, self.classes))  
        class_name_list = []
        instance_list = []
        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                list(range(len(self.json_data[class_name]))),
                self.K + self.Q, False)  
            count = 0
            class_name_list.append(class_name)
            instance_list.append(list(indices))
            for j in indices:  
                word, pos1, pos2, mask, pos1_end, pos2_end = self.__getraw__(
                    self.json_data[class_name][j])  
                word = torch.tensor(word).long()  
                pos1 = torch.tensor(pos1).long()
                pos2 = torch.tensor(pos2).long()
                mask = torch.tensor(mask).long()
                pos1_end = torch.tensor(pos1_end).long()
                pos2_end = torch.tensor(pos2_end).long()
                if count < self.K:
                    self.__additem__(support_set, word, pos1, pos2, mask, pos1_end, pos2_end)  
                else:
                    self.__additem__(query_set, word, pos1, pos2, mask, pos1_end, pos2_end)  
                count += 1

            query_label += [i] * self.Q  

        id_set['class_name'].append(class_name_list)
        id_set['instance'].append(instance_list)

        # NA
        for j in range(Q_na):  
            cur_class = np.random.choice(na_classes, 1, False)[0]  
            index = np.random.choice(
                list(range(len(self.json_data[cur_class]))),
                1, False)[0]  
            word, pos1, pos2, mask, pos1_end, pos2_end = self.__getraw__(
                self.json_data[cur_class][index])  
            word = torch.tensor(word).long()
            pos1 = torch.tensor(pos1).long()
            pos2 = torch.tensor(pos2).long()
            mask = torch.tensor(mask).long()
            pos1_end = torch.tensor(pos1_end).long()
            pos2_end = torch.tensor(pos2_end).long()
            # sen = torch.tensor(sen).long()
            self.__additem__(query_set, word, pos1, pos2, mask, pos1_end, pos2_end)  
        query_label += [self.N] * Q_na  

        return support_set, query_set, query_label, id_set, target_classes

    def __len__(self):
        return 1000000000


def collate_fn(data):
    batch_support = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'pos1_end': [], 'pos2_end': []}
    batch_query = {'word': [], 'pos1': [], 'pos2': [], 'mask': [], 'pos1_end': [], 'pos2_end': []}
    batch_label = []
    batch_class = []
    batch_id = {'class_name': [], 'instance': []}
    support_sets, query_sets, query_labels, id_sets, classes = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:  
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        batch_label += query_labels[i]
        batch_class += classes[i]
        for k in id_sets[i]:
            batch_id[k] += id_sets[i][k]  
    for k in batch_support:
        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_support, batch_query, batch_label, batch_id, batch_class


def get_loader(name, encoder, N, K, Q, batch_size,
               num_workers=8, collate_fn=collate_fn, na_rate=0, root='./data'):
    dataset = FewRelDataset(name, encoder, N, K, Q, na_rate, root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return iter(data_loader)

class FewRelDatasetPair(data.Dataset):
    '''
    bridge pair Dataset
    '''
    def __init__(self, name, encoder, N, K, Q, na_rate, root, encoder_name):
        self.root = root
        path = os.path.join(root, name + ".json")  
        if not os.path.exists(path):
            print(path)
            print("[ERROR] Data file does not exist!")
            assert (0)
        self.json_data = json.load(open(path))
        self.classes = list(self.json_data.keys())  
        self.N = N
        self.K = K
        self.Q = Q
        self.na_rate = na_rate
        self.encoder = encoder
        self.encoder_name = encoder_name
        self.max_length = encoder.max_length  

    def __getraw__(self, item):
        word = self.encoder.tokenize(item['tokens'], item['h'][1], item['t'][1])  
        return word  

    def __additem__(self, d, word, pos1, pos2, mask):
        d['word'].append(word)
        d['pos1'].append(pos1)
        d['pos2'].append(pos2)
        d['mask'].append(mask)

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)  
        support = []
        query = []
        fusion_set = {'word':[], 'mask':[], 'seg':[]}
        query_label = []
        Q_na = int(self.na_rate * self.Q) 
        na_classes = list(filter(lambda x: x not in target_classes, self.classes))  


        for i, class_name in enumerate(target_classes):  
            indices = np.random.choice(list(range(len(self.json_data[class_name]))), self.K + self.Q, False)  
            count = 0
            for j in indices:
                word = self.__getraw__(self.json_data[class_name][j])  
                if count < self.K:
                    support.append(word)  
                else:
                    query.append(word)  
                count += 1

            query_label += [i] * self.Q  

        # NA
        for j in range(Q_na):
            cur_class = np.random.choice(na_classes, 1, False)[0]  
            index = np.random.choice(list(range(len(self.json_data[cur_class]))), 1, False)[0]
            word = self.__getraw__(self.json_data[cur_class][index])
            query_label += [self.N] * Q_na  


        # pair
        for word_query in query:
            for word_support in support:
                if self.encoder_name == 'bert':
                    SEP = self.encoder.tokenizer.convert_tokens_to_ids(['[SEP]'])
                    CLS = self.encoder.tokenizer.convert_tokens_to_ids(['[CLS]'])
                    word_tensor = torch.zeros((self.max_length)).long()
                else:
                    SEP = self.encoder.tokenizer.convert_tokens_to_ids(['</s>'])
                    CLS = self.encoder.tokenizer.convert_tokens_to_ids(['<s>'])
                    word_tensor = torch.zeros((self.max_length)).long()
                new_word = CLS + word_support + SEP + word_query + SEP  
                for i in range(min(self.max_length, len(new_word))):
                    word_tensor[i] = new_word[i]  
                mask_tensor = torch.zeros((self.max_length)).long()
                mask_tensor[:min(self.max_length, len(new_word))] = 1
                seg_tensor = torch.ones((self.max_length)).long()
                seg_tensor[:min(self.max_length, len(word_support) + 1)] = 0
                fusion_set['word'].append(word_tensor)
                fusion_set['mask'].append(mask_tensor)
                fusion_set['seg'].append(seg_tensor)

        return fusion_set, query_label

    def __len__(self):
        return 1000000000


def collate_fn_pair(data):
    batch_set = {'word': [], 'seg': [], 'mask': []}
    batch_label = []
    fusion_sets, query_labels = zip(*data)
    
    for i in range(len(fusion_sets)):
        for k in fusion_sets[i]:
            batch_set[k] += fusion_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_set:
        batch_set[k] = torch.stack(batch_set[k], 0)
    batch_label = torch.tensor(batch_label)
    return batch_set, batch_label


def get_loader_pair(name, encoder, N, K, Q, batch_size,
        num_workers=8, collate_fn=collate_fn_pair, na_rate=0, root='./data/test_sf', encoder_name='bert'):
    dataset = FewRelDatasetPair(name, encoder, N, K, Q, na_rate, root, encoder_name)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)  
    return iter(data_loader)
