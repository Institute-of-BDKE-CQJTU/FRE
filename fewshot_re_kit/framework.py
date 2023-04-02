import os
import random
import sys
import torch
from torch import optim, nn
from collections import Counter
from transformers import AdamW, get_linear_schedule_with_warmup

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def warmup_linear(global_step, warmup_step):
    if global_step < warmup_step:
        return global_step / warmup_step
    else:
        return 1.0

class FewShotREModel(nn.Module):
    def __init__(self, my_sentence_encoder):
        nn.Module.__init__(self)
        self.sentence_encoder = nn.DataParallel(my_sentence_encoder, device_ids=[0])  

        self.cost = nn.CrossEntropyLoss()

    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        return: logits, pred
        '''
        raise NotImplementedError

    def loss(self, logits, label):
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1)) 

    def accuracy(self, pred, label):
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))  

class FewShotREFramework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader, adv_data_loader=None, adv=False, d=None):
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.adv_data_loader = adv_data_loader
        self.adv = adv
        if adv:
            self.adv_cost = nn.CrossEntropyLoss()
            self.d = d
            self.d.cuda()

    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint  
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)

    def item(self, x):
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def train(self, model, model_name, B, N_for_train, N_for_eval, K, Ke, Q, na_rate=0, learning_rate=1e-1,
              lr_step_size=20000,
              weight_decay=1e-5,
              train_iter=30000,
              val_iter=1000,
              val_step=2000,
              test_iter=3000,
              load_ckpt=None,
              save_ckpt=None,
              pytorch_optim=optim.SGD,
              bert_optim=False,
              warmup=True,
              warmup_step=300,
              grad_iter=1,
              fp16=False,
              pair=False,
              adv_dis_lr=1e-1,
              adv_enc_lr=1e-1,
              use_sgd_for_bert=False):
        print('Start training......')

        random.seed(42)
        torch.manual_seed(42)

        if bert_optim:
            print('Use bert optim!')
            parameters_to_optimize = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            parameters_to_optimize = [{'params': [p for n, p in parameters_to_optimize
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in parameters_to_optimize
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
            if use_sgd_for_bert:
                optimizer = torch.optim.SGD(parameters_to_optimize, lr=learning_rate)
            else:
                optimizer = AdamW(parameters_to_optimize, lr=learning_rate, correct_bias=False)
            if self.adv:
                optimizer_encoder = AdamW(parameters_to_optimize, lr=1e-5, correct_bias=False)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter)
        else:
            optimizer = pytorch_optim(model.parameters(), learning_rate, weight_decay=weight_decay)
            if self.adv:
                optimizer_encoder = pytorch_optim(model.parameters(), lr=adv_enc_lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size)
        if self.adv:
            optimizer_dis = pytorch_optim(self.d.parameters(), lr=adv_dis_lr)

        if load_ckpt:
            state_dict = self.__load_model__(load_ckpt)['state_dict']
            own_state = model.state_dict()
            for name, param in state_dict.items():  
                if name not in own_state:
                    print('ignore {}'.format(name))
                    continue
                print('load {} from {}'.format(name, load_ckpt))
                own_state[name].copy_(param)
            start_iter = 0
        else:
            start_iter = 0

        if fp16:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

        model.train()
        if self.adv:
            self.d.train()

        # training
        best_acc = 0
        iter_loss = 0.0
        iter_loss_dis = 0.0
        iter_right = 0.0
        iter_right_dis = 0.0
        iter_sample = 0.0
        for it in range(start_iter, start_iter + train_iter):  # epoch
            support, query, label, ids, classes = next(self.train_data_loader)
            if torch.cuda.is_available():
                for k in support:
                    support[k] = support[k].cuda()
                for k in query:
                    query[k] = query[k].cuda()
                label = label.cuda()
                # ids = ids.cuda()
                # logits, pred, s_loss = model(support, query, N_for_train, K, Q * N_for_train + na_rate * Q)
                logits, pred, loss_class = model(support, query, N_for_train, K, Q * N_for_train + na_rate * Q)
                # logits, pred = model(support, query, N_for_train, K, Q * N_for_train + na_rate * Q)
            loss = model.loss(logits, label) / float(grad_iter)
            loss = loss + loss_class
            right = model.accuracy(pred, label)

            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)  

            if it % grad_iter == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            iter_loss += self.item(loss.data)
            iter_right += self.item(right.data)
            iter_sample += 1
            if self.adv:
                sys.stdout.write(
                    'step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%, dis_loss: {3:2.6f}, dis_acc: {4:2.6f}'
                    .format(it + 1, iter_loss / iter_sample,
                            100 * iter_right / iter_sample,
                            iter_loss_dis / iter_sample,
                            100 * iter_right_dis / iter_sample) + '\r')
            else:
                sys.stdout.write(
                    'step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample,
                                                                               100 * iter_right / iter_sample) + '\r')
            sys.stdout.flush()

            if (it + 1) % val_step == 0:
                acc = self.eval(model, B, N_for_eval, Ke, Q, val_iter, na_rate=na_rate, pair=pair)
                model.train()
                if acc > best_acc:
                    print('Best checkpoint')
                    torch.save({'state_dict': model.state_dict()}, save_ckpt)
                    best_acc = acc
                iter_loss = 0
                iter_loss_dis = 0
                iter_right = 0
                iter_right_dis = 0
                iter_sample = 0

        print('Finish training ' + model_name)

    def eval(self, model, B, N, Ke, Q, eval_iter, na_rate=0, pair=False, ckpt=None):
        print('')

        model.eval()
        if ckpt is None:
            print('Use val dataset')
            eval_dataset = self.val_data_loader
        else:
            print('Use test dataset')
            if ckpt != 'None':
                state_dict = self.__load_model__(ckpt)['state_dict']
                own_state = model.state_dict()
                for name, param in state_dict.items():
                    if name not in own_state:
                        continue
                    own_state[name].copy_(param)
            eval_dataset = self.test_data_loader

        iter_right = 0.0
        iter_loss = 0.0
        iter_sample = 0.0
        count_class_neg = []
        with torch.no_grad():
            for it in range(eval_iter):
                support, query, label, ids, classes = next(eval_dataset)
                if torch.cuda.is_available():
                    for k in support:
                        support[k] = support[k].cuda()
                    for k in query:
                        query[k] = query[k].cuda()
                    label = label.cuda()
                    # ids = ids.cuda()
                # logits, pred = model(support, query, N, K, Q * N + Q * na_rate)
                logits, pred, loss_out = model(support, query, N, Ke, Q * N + Q * na_rate)
                loss = model.loss(logits, label)/float(1)
                loss = loss + loss_out
                for i, cla in enumerate(pred):
                    if cla != label[i]:
                        count_class_neg.append(classes[i])

                right = model.accuracy(pred, label)
                iter_right += self.item(right.data)
                iter_loss += self.item(loss.data)
                iter_sample += 1

                sys.stdout.write('[EVAL] step: {0:4} | loss: {1:2.6f}, accuracy: {2:3.2f}%'.format(it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample) + '\r')
                sys.stdout.flush()

            # count = Counter(count_class_neg)
            # sys.stdout.write('\n' + 'count:{}'.format(count))
            print('')
        return iter_right / iter_sample
