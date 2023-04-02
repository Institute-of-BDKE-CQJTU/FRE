from fewshot_re_kit.data_loader import get_loader_pair, get_loader
from fewshot_re_kit.framework import FewShotREFramework
from fewshot_re_kit.sentence_encoder import BERTSentenceEncoder, RoBERTaSentenceEncoder
from model.proto import Proto
import sys
import torch
from torch import optim, nn
import numpy as np
import json
import argparse
import os
import random

#yiliao_train  bridge-train_have_negative
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='yiliao_train_for_DD',
                        help='train file')  
    parser.add_argument('--val', default='yiliao_test_for_DD',
                        help='val file')
    parser.add_argument('--test', default='yiliao_test_for_DD',
                        help='test file')
    parser.add_argument('--adv', default=None,
                        help='adv file')
    parser.add_argument('--trainN', default=5, type=int,
                        help='N in train')
    parser.add_argument('--N', default=5, type=int,
                        help='N way')
    parser.add_argument('--K', default=1, type=int,
                        help='K shot')
    parser.add_argument('--Ke', default=1, type=int,
                        help='K shot in eval')
    parser.add_argument('--Q', default=1, type=int,
                        help='Num of query per class')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='batch size')
    parser.add_argument('--train_iter', default=8000, type=int,
                        help='num of iters in training')
    parser.add_argument('--val_iter', default=500, type=int,
                        help='num of iters in validation')
    parser.add_argument('--test_iter', default=5000, type=int,
                        help='num of iters in testing')
    parser.add_argument('--val_step', default=500, type=int,
                        help='val after training how many iters')
    parser.add_argument('--model', default='proto',
                        help='model name')
    parser.add_argument('--encoder', default='roberta',
                        help='encoder: cnn or bert or roberta')
    parser.add_argument('--max_length', default=256, type=int,
                        help='max length')
    parser.add_argument('--lr', default=1e-5, type=float,
                        help='learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                        help='weight decay')
    parser.add_argument('--dropout', default=0.0, type=float,
                        help='dropout rate')
    parser.add_argument('--na_rate', default=0, type=int,
                        help='NA rate (NA = Q * na_rate)')
    parser.add_argument('--grad_iter', default=1, type=int,
                        help='accumulate gradient every x iterations')
    parser.add_argument('--optim', default='adam',
                        help='sgd / adam / adamw')
    parser.add_argument('--hidden_size', default=768, type=int,
                        help='hidden size')
    parser.add_argument('--load_ckpt', default=None,
                        help='load ckpt')
    parser.add_argument('--save_ckpt', default=None,
                        help='save ckpt')
    parser.add_argument('--fp16', action='store_true',
                        help='use nvidia apex fp16')
    parser.add_argument('--only_test', action='store_true',
                        help='only test')
    parser.add_argument('--ckpt_name', type=str, default='',
                        help='checkpoint name.')

    # only for bert / roberta
    parser.add_argument('--pair', action='store_true',
                        help='use pair model')
    parser.add_argument('--pretrain_ckpt', default=None,
                        help='bert / roberta pre-trained checkpoint')
    parser.add_argument('--cat_entity_rep', action='store_true',
                        help='concatenate entity representation as sentence rep')

    # only for prototypical networks
    parser.add_argument('--dot', action='store_true',
                        help='use dot instead of L2 distance for proto')

    # only for mtb
    parser.add_argument('--no_dropout', action='store_true',
                        help='do not use dropout after BERT (still has dropout in BERT).')

    # experiment
    parser.add_argument('--mask_entity', action='store_true',
                        help='mask entity names')
    parser.add_argument('--use_sgd_for_bert', action='store_true',
                        help='use SGD instead of AdamW for BERT.')
    parser.add_argument('--pretrain_model', type=str, default='./pretrain/RoBERTa_zh_L12_PyTorch', help='pretrained model'
                                                                                                     'select')
    parser.add_argument('--cat', type=str, default='', help='if cat entity.')
    parser.add_argument('--iter', type=int, default='')

    opt = parser.parse_args() 
    trainN = opt.trainN
    N = opt.N
    K = opt.K
    Ke = opt.Ke
    Q = opt.Q
    batch_size = opt.batch_size
    model_name = opt.model
    encoder_name = opt.encoder
    max_length = opt.max_length
    pretrain_model = opt.pretrain_model
    cat = opt.cat

    print("{}-way-{}-shot Few-Shot Relation Classification".format(N, K))
    print("model: {}".format(model_name))
    print("encoder: {}".format(encoder_name))
    print("max_length: {}".format(max_length))
    print('pretrained model: {}'.format(pretrain_model))

    if encoder_name == 'bert':
        pretrain_ckpt = pretrain_model
        sentence_encoder = BERTSentenceEncoder(
            pretrain_ckpt,
            max_length,
            cat_entity_rep=opt.cat_entity_rep,
            mask_entity=opt.mask_entity)
    elif encoder_name == 'roberta':
        pretrain_ckpt = pretrain_model
        sentence_encoder = RoBERTaSentenceEncoder(
            pretrain_ckpt,
            max_length,
            cat_entity_rep=opt.cat_entity_rep,
            mask_entity=opt.mask_entity)
    else:
        raise NotImplementedError

    train_data_loader = get_loader(opt.train, sentence_encoder,
                                   N=trainN, K=K, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
    val_data_loader = get_loader(opt.val, sentence_encoder,
                                 N=N, K=Ke, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)
    test_data_loader = get_loader(opt.test, sentence_encoder,
                                  N=N, K=Ke, Q=Q, na_rate=opt.na_rate, batch_size=batch_size)

    if opt.optim == 'sgd':
        pytorch_optim = optim.SGD
    elif opt.optim == 'adam':
        pytorch_optim = optim.Adam
    elif opt.optim == 'adamw':
        from transformers import AdamW
        pytorch_optim = AdamW
    else:
        raise NotImplementedError
    framework = FewShotREFramework(train_data_loader, val_data_loader, test_data_loader)
    prefix = '-'.join([model_name, encoder_name, opt.optim, opt.train, opt.val, str(N), str(K), str(Ke), cat, str(opt.iter)])
    if opt.adv is not None:
        prefix += '-adv_' + opt.adv
    if opt.na_rate != 0:
        prefix += '-na{}'.format(opt.na_rate)
    if opt.dot:
        prefix += '-dot'
    if opt.cat_entity_rep:
        prefix += '-catentity'
    if len(opt.ckpt_name) > 0:
        prefix += '-' + opt.ckpt_name

    if model_name == 'proto':
        model = Proto(sentence_encoder, dot=opt.dot)
    else:
        raise NotImplementedError

    if not os.path.exists('checkpoint/x'):
        os.mkdir('checkpoint/x')
    if pretrain_model == './pretrain/chinese-bert-wwm-ext':
        ckpt = 'checkpoint/x/{}.pth.tar'.format(prefix)
    elif pretrain_model == './pretrain/RoBERTa_zh_L12_PyTorch':
        ckpt = 'checkpoint/x/{}.RoBERTa_zh_L12_PyTorch.pth.tar'.format(prefix)
    if opt.save_ckpt:
        ckpt = opt.save_ckpt

    if torch.cuda.is_available():
        model.cuda()

    if not opt.only_test:
        if encoder_name in ['bert', 'roberta']:
            bert_optim = True
        else:
            bert_optim = False

        if opt.lr == -1:
            if bert_optim:
                opt.lr = 2e-5
            else:
                opt.lr = 1e-1

        opt.train_iter = opt.train_iter * opt.grad_iter
        framework.train(model, prefix, batch_size, trainN, N, K, Ke, Q,
                        pytorch_optim=pytorch_optim, load_ckpt=opt.load_ckpt, save_ckpt=ckpt,
                        na_rate=opt.na_rate, val_step=opt.val_step, fp16=opt.fp16, pair=opt.pair,
                        train_iter=opt.train_iter, val_iter=opt.val_iter, bert_optim=bert_optim,
                        learning_rate=opt.lr, use_sgd_for_bert=opt.use_sgd_for_bert, grad_iter=opt.grad_iter)
    else:
        ckpt = opt.load_ckpt
        if ckpt is None:
            print("Warning: --load_ckpt is not specified. Will load Hugginface pre-trained checkpoint.")
            ckpt = 'none'

    acc = framework.eval(model, batch_size, N, Ke, Q, opt.test_iter, na_rate=opt.na_rate, ckpt=ckpt, pair=opt.pair)
    print("RESULT: %.2f" % (acc * 100))

    if not os.path.exists('result'):
        os.mkdir('result')
    with open('./result/result.txt', 'a', encoding='utf-8') as f1:
        f1.write(prefix + ':' + '\t')
        f1.write(str(acc * 100) + '\n')


if __name__ == '__main__':
    main()
