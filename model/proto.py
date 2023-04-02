import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import nn
import torch.nn.functional as F

class Proto(fewshot_re_kit.framework.FewShotREModel):

    def __init__(self, sentence_encoder, dot=False):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.drop = nn.Dropout(0.2)
        self.dot = dot
        self.lstm = nn.LSTM(input_size=768, hidden_size=768, batch_first=True, dropout=0.3, bidirectional=True)
        self.fc1 = nn.Linear(768 * 2, 768)
        self.fc = nn.Linear(768 * 2, 768 * 2)
        self.tanh = nn.Tanh()
        self.combiner = "max"


    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)  

    def __batch_dist__(self, S, Q):
        return self.__dist__(S, Q.unsqueeze(2), 3)  

    def __cat__(self, a, b, Q):
        tensor_range = a.size()[0]
        total_q_range = a.size()[1]
        hidden_size = int((a.size()[2]/3)*5)
        type_range = b.size()[1]
        query_type = torch.zeros(tensor_range, total_q_range, hidden_size)
        for i in torch.arange(tensor_range):
            for j in torch.arange(type_range):
                n = j*Q
                for m in torch.arange(n, n+Q):
                    query_type[i][m] = torch.cat((a[i][m], b[i][j]), 0)
        query_type = query_type.cuda()
        return query_type


    def loss_out_class(self, classproto):
        total_loss = 0.0
        loss = 0.0
        N = classproto.size()[1]
        for i in range(classproto.size()[0]):
            for j in range(classproto.size()[1]):
                p1 = classproto[i][j]
                for k in range(classproto.size()[1]):
                    if k != j:
                        iter1 = pow(p1 - classproto[i][k], 2)
                        iter1 = torch.sum(iter1)
                        loss = torch.sqrt(iter1)
                        total_loss = total_loss + loss
                    else:
                        loss = loss
                        total_loss = total_loss
        total_loss = total_loss/2.0
        total_loss = total_loss/float(pow(N, 2))
        total_loss = 1 - total_loss

        loss_out = max(0, total_loss)
        return loss_out


    def loss_out_class_att(self, classproto):
        total_loss = 0.0
        loss = 0.0
        N = classproto.size()[2]
        for i in range(classproto.size()[0]):
            for j in range(classproto.size()[1]):
                for n in range(N):
                    p1 = classproto[i][j][n]
                    for k in range(classproto.size()[2]):
                        if k != n:
                            iter1 = pow(p1 - classproto[i][j][k], 2)
                            iter1 = torch.sum(iter1)
                            loss = torch.sqrt(iter1)
                            total_loss = total_loss + loss
                        else:
                            continue
        total_loss = total_loss / 2.0
        total_loss = total_loss / float(pow(N, 2))
        total_loss = 1 - total_loss

        loss_out = max(0, total_loss)
        return loss_out


    def forward(self, support2, query, N, K, total_Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''

        support_emb, ent_ss_emb, ent_se_emb, sten_out = self.sentence_encoder(support2)  # (B * N * K, D), where D is the hidden size
        query_emb, ent_qs_emb, ent_qe_emb, sten_out_q = self.sentence_encoder(query)  # (B * total_Q, D)
        # support_emb = self.tanh(support_emb)
        # query_emb = self.tanh(query_emb)
        hidden_size = support_emb.size(-1)  # 768
        """********************************"""
        support_emb = self.drop(support_emb)
        query_emb = self.drop(query_emb)
        """********************************"""
        H, (h_n, c_n) = self.lstm(support_emb)
        out = torch.cat((h_n[-2], h_n[-1]), 1)
        out = self.fc1(out)
        out = self.tanh(out)

        H_q, (h_n_q, c_n_q) = self.lstm(query_emb)
        out_q = torch.cat((h_n_q[-2], h_n_q[-1]), 1)
        out_q = self.fc1(out_q)
        out_q = self.tanh(out_q)

        ent_s_emb = torch.cat((ent_ss_emb, ent_se_emb), 1)  #（B*N*K, D）
        ent_q_emb = torch.cat((ent_qs_emb, ent_qe_emb), 1)  #（B*N*K, D）

        support = torch.cat((out, ent_s_emb), 1)
        query = torch.cat((out_q, ent_q_emb), 1)

        support = support.view(-1, N, K, 2*hidden_size)  # (B, N, K, 2D)
        query = query.view(-1, total_Q, 2*hidden_size)  # (B, total_Q, 2D)

        support = support.unsqueeze(1).expand(-1, total_Q, -1, -1, -1)  # (B, Q, N, K, D)
        support_for_att = self.fc(support)
        query_for_att = self.fc(query.unsqueeze(2).unsqueeze(3).expand(-1, -1, N, K, -1))
        ins_att_score = F.softmax(torch.tanh(support_for_att * query_for_att).sum(-1), dim=-1)  # (B, total_Q, N, K)
        support_proto = (support * ins_att_score.unsqueeze(4).expand(-1, -1, -1, -1, 2*hidden_size)).sum(3)  # (B, total_Q, N, 2D)

        loss_class = self.loss_out_class_att(support_proto)

        logits = self.__batch_dist__(support_proto, query)
        _, pred = torch.max(logits.view(-1, N), 1)
        return logits, pred, loss_class
