
# -*- coding: utf8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from torch.utils.data import Dataset
import re

import random
import logging
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math


import sys

import argparse
import numpy as np

from os import path

from distutils.util import strtobool

from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


logger = logging.getLogger()

class CodeDataset(Dataset):
    """dataset for training/test"""

    def __init__(self, starts, paths, ends, labels, transform=None):
        # self.ids = ids
        self.starts = starts
        self.paths = paths
        self.ends = ends
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        item = {
            # 'id': self.ids[index],
            'starts': self.starts[index],
            'paths': self.paths[index],
            'ends': self.ends[index],
            'label': self.labels[index]
        }
        if self.transform:
            item = self.transform(item)
        return item

class CodeData(object):
    """data corresponding to one method"""

    def __init__(self):
        self.id = None
        self.label = None
        self.path_contexts = []
        self.source = None

# -*- coding: utf8 -*-


class DatasetBuilder(object):
    """transform dataset for training and test"""

    def __init__(self, reader, option, split_ratio=0.2):
        self.reader = reader
        self.option = option
        
        test_count = int(len(reader.items) * 0.1)
        val_count = int(len(reader.items) * split_ratio+0.1)
        random.shuffle(reader.items)
        # print(reader.items[0][0].path_contexts[0])
        train_items = reader.items[val_count:]
        val_items = reader.items[test_count:val_count]
        test_items = reader.items[0:test_count]
        logger.info('train item size: {0}'.format(len(train_items)))
        logger.info('test item size: {0}'.format(len(test_items)))

        train_dataset_size = 0
        test_dataset_size = 0
        val_dataset_size = 0
        # logger.info('train dataset size: {0}'.format(train_dataset_size))
        # logger.info('test dataset size: {0}'.format(test_dataset_size))

        self.train_items = train_items
        self.test_items = test_items
        self.val_items = val_items
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None

    def refresh_train_dataset(self):
        """refresh training dataset (shuffling path contexts and picking up items (#items <= max_path_length)"""
        inputs_starts, inputs_paths, inputs_ends, inputs_label = self.build_data(self.reader, self.train_items, self.option.max_path_length)
        self.train_dataset = CodeDataset(inputs_starts, inputs_paths, inputs_ends, inputs_label)
    def refresh_val_dataset(self):
        """refresh training dataset (shuffling path contexts and picking up items (#items <= max_path_length)"""
        inputs_starts, inputs_paths, inputs_ends, inputs_label = self.build_data(self.reader, self.val_items, self.option.max_path_length)
        self.val_dataset = CodeDataset(inputs_starts, inputs_paths, inputs_ends, inputs_label)
    def refresh_test_dataset(self):
        """refresh test dataset (shuffling path contexts and picking up items (#items <= max_path_length)"""
        inputs_starts, inputs_paths, inputs_ends, inputs_label = self.build_data(self.reader, self.test_items, self.option.max_path_length)
        self.test_dataset = CodeDataset(inputs_starts, inputs_paths, inputs_ends, inputs_label)

    def build_data(self, reader, items, max_path_length):


        inputs_label = []
        i=1
        fcount = self.option.fcount
        input_starts = [[] for i in range(fcount)]
        input_paths = [[] for i in range(fcount)]
        input_ends = [[] for i in range(fcount)]

        question_token_index = 1
        # replace @method_0 with @question
        # method_token_index = terminal_vocab_stoi["@method_0"]

        for item in items:
            # inputs_id.append(item.id)
            # label_index = label_vocab_stoi[item.normalized_label]
            # print(len(item))
            inputs_label.append(item[0].label)


            # random.shuffle(item.path_contexts)
            for i in range(fcount):
              starts = []
              paths = []
              ends = []      
              for m in range(min(max_path_length,len(item[i].path_contexts))):
                  
                  start, path, end  = item[i].path_contexts[m]
                  # if start == method_token_index:
                  #     start = question_token_index
                  starts.append(start)
                  paths.append(path)
                  # if end == method_token_index:
                  #     end = question_token_index
                  ends.append(end)
              # print(len(starts))
              starts = self.pad_inputs(starts, max_path_length)
              paths = self.pad_inputs(paths, max_path_length)
              ends = self.pad_inputs(ends, max_path_length)
              input_starts[i].append(starts)
              input_paths[i].append(paths)
              input_ends[i].append(ends)
              # print(len(starts))
        for c in range(fcount):
          input_starts[c] = torch.tensor(input_starts[c], dtype=torch.long)
          input_paths[c] = torch.tensor(input_paths[c], dtype=torch.long)
          input_ends[c] = torch.tensor(input_ends[c], dtype=torch.long)
        inputs_label = torch.tensor(inputs_label, dtype=torch.long)
        # temp = torch.split(input_starts[0],4,0)
        # print("starts "+str(temp[0].size()))
        # print("label"+str(len(inputs_label)))
        return input_starts, input_paths, input_ends, inputs_label

    def pad_inputs(self, data, length, pad_value=0):
        """pad values"""

        assert len(data) <= length

        count = length - len(data)
        data.extend([pad_value] * count)
        return data


QUESTION_TOKEN_INDEX = 1
QUESTION_TOKEN_NAME = "@question"

class DatasetReader(object):
    """read dataset file"""

    def __init__(self, corpus_path):
        self.pad_data = CodeData()
        self.pad_data.path_contexts.append((0,0,0))
        self.items = []
        self.load(corpus_path)
        self.round = 0
        
        logger.info('corpus: {0}'.format(len(self.items)))


    def load(self, corpus_path):
        i = 0
        with open(corpus_path, mode="r", encoding="utf-8") as f:
            code_data = None
            path_contexts_append = None
            parse_mode = 0
            funcItem = []
            # label_vocab = self.label_vocab
            # label_vocab_append = label_vocab.append
            t = 0
            for line in f.readlines():

                if line.startswith('R'):
                    self.round = int(line[1:])+1
                    self.pad_data.label = self.round
                    # if code_data is None:
                    #   code_data = CodeData()
                    #   path_contexts_append = code_data.path_contexts.append
                    # parse_mode = 0
                elif line.startswith('File'):
                    if(t != 0):
                      if code_data is not None:
                          funcItem.append(code_data)
                          # print(code_data.path_contexts[0])
                          code_data = None
                      for p in range(8-len(funcItem)):
                        funcItem.append(self.pad_data)
                      self.items.append(funcItem)
                      funcItem = []
                      # if code_data is None:
                      #   code_data = CodeData()
                      #   path_contexts_append = code_data.path_contexts.append
                    # parse_mode = 0
                elif line.startswith("function"):
                    if code_data is not None:
                          funcItem.append(code_data)
                          # print(code_data.path_contexts[0])
                          code_data = None
                          
                    if code_data is None:
                      code_data = CodeData()
                      path_contexts_append = code_data.path_contexts.append
                    code_data.label = self.round
                    # print(code_data.path_contexts)
                    parse_mode = 1
                    i = 0
                    t = 1
                elif parse_mode == 1:
                    if(line != '\n'):
                      path_context = line.split('\t')
                      if(i<1000):
                        path_contexts_append((int(path_context[0]) + QUESTION_TOKEN_INDEX,
                                            int(path_context[1]),
                                            int(path_context[2]) + QUESTION_TOKEN_INDEX))
                        # print(path_context[0])
                        i+=1


NINF = - 3.4 * math.pow(10, 38)  # -Inf

class Code2Vec(nn.Module):
    """the code2vec model"""

    def __init__(self, option):
        super(Code2Vec, self).__init__()
        self.option = option
        self.terminal_embedding = nn.Embedding(option.terminal_count, option.terminal_embed_size)
        self.path_embedding = nn.Embedding(option.path_count, option.path_embed_size)
        self.input_linear = nn.Linear(option.terminal_embed_size * 2 + option.path_embed_size, option.encode_size, bias=False)
        self.input_layer_norm = nn.LayerNorm(option.encode_size)

        if 0.0 < option.dropout_prob < 1.0:
            self.input_dropout = nn.Dropout(p=option.dropout_prob)
        else:
            self.input_dropout = None
        self.attention_parameter = [None]*(option.fcount+1)
        for j in range(option.fcount+1):
          self.attention_parameter[j] = Parameter(torch.nn.init.xavier_normal_(torch.zeros(option.encode_size, 1, dtype=torch.float32, requires_grad=True)).view(-1), requires_grad=True)

        if option.angular_margin_loss:
            self.output_linear = Parameter(torch.FloatTensor(option.label_count, option.encode_size))
            nn.init.xavier_uniform_(self.output_linear)
            self.cos_m = math.cos(option.angular_margin)
            self.sin_m = math.sin(option.angular_margin)
            self.th = math.cos(math.pi - option.angular_margin)
            self.mm = math.sin(math.pi - option.angular_margin) * option.angular_margin
        else:
            self.output_linear = nn.Linear(option.encode_size, option.label_count, bias=True)

            self.output_linear.bias.data.fill_(0.0)

    def forward(self, starts, paths, ends, label):
        option = self.option
        # embedding
        code_vectors = []
        for i in range(option.fcount):
          embed_starts = self.terminal_embedding(starts[i])
          embed_paths = self.path_embedding(paths[i])
          embed_ends = self.terminal_embedding(ends[i])

          combined_context_vectors = torch.cat((embed_starts, embed_paths, embed_ends), dim=2)
          # FNN, Layer Normalization, tanh
          combined_context_vectors = self.input_linear(combined_context_vectors)

          ccv_size = combined_context_vectors.size()
          combined_context_vectors = self.input_layer_norm(combined_context_vectors.view(-1, option.encode_size)).view(ccv_size)
          combined_context_vectors = torch.tanh(combined_context_vectors)
          # dropout
          if self.input_dropout is not None:
              combined_context_vectors = self.input_dropout(combined_context_vectors)

          attn_mask = (starts[i] > 0).float()
          attention = self.get_attention(combined_context_vectors, attn_mask,i)

        # code vector
          expanded_attn = attention.unsqueeze(-1).expand_as(combined_context_vectors)
          code_vector = torch.sum(torch.mul(combined_context_vectors, expanded_attn), dim=1)
          code_vector = code_vector.tolist()
          temp = []
          for h in range(len(code_vector)): 
            temp.append([code_vector[h]])
          code_vectors.append(temp)
        get_cv = torch.tensor(code_vectors[0])
        for ind in range(1,len(code_vectors)):
          get_cv = torch.cat((get_cv,torch.tensor(code_vectors[ind])),dim = 1)
        
        fattn_mask = (get_cv.mean(2) > 0).float()

        fattention = self.get_attention(get_cv, fattn_mask,8)

        # code vector
        fexpanded_attn = fattention.unsqueeze(-1).expand_as(get_cv)
        fcode_vector = torch.sum(torch.mul(get_cv, fexpanded_attn), dim=1)  
        if option.angular_margin_loss:
            # angular margin loss
            cosine = F.linear(F.normalize(code_vector), F.normalize(self.output_linear))
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            phi = cosine * self.cos_m - sine * self.sin_m
            phi = torch.where(cosine > 0, phi, cosine)
            one_hot = torch.zeros(cosine.size(), device=option.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            outputs = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            outputs *= option.inverse_temp
        else:
            outputs = self.output_linear(fcode_vector)

        return outputs, fcode_vector, attention

    def get_attention(self, vectors, mask,num):
        """calculate the attention of the (masked) context vetors. mask=1: meaningful value, mask=0: padded."""
        expanded_attn_param = self.attention_parameter[num].unsqueeze(0).expand_as(vectors)
        attn_ca = torch.mul(torch.sum(vectors * expanded_attn_param, dim=2), mask) + (1 - mask) * NINF
        attention = F.softmax(attn_ca, dim=1)
        return attention


def batch_data(data,batch_size,fcount):
  batched_data = []
  actual_starts = []
  actual_paths = []
  actual_ends = []
  for i in range(fcount):
    actual_starts.append(torch.split(data.starts[i],batch_size,0))
    actual_paths.append(torch.split(data.paths[i],batch_size,0))
    actual_ends.append(torch.split(data.ends[i],batch_size,0))
  actual_labels = torch.split(data.labels,batch_size,0)
  for i in range(len(actual_starts[0])):
    starts = []
    paths = []
    ends = []
    for j in range(fcount):
      starts.append(actual_starts[j][i])
      paths.append(actual_paths[j][i])
      ends.append(actual_ends[j][i])
    batched_data.append(CodeDataset(starts,paths,ends,actual_labels[i]))
  return batched_data

# -*- coding: utf8 -*-


sys.path.append('.')


logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: %(message)s', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)
random_seed=123

corpus_path="corpus.txt"

batch_size=32
terminal_embed_size=100
path_embed_size=100
encode_size=300
max_path_length=200
model_path="./"
vectors_path="code.vec"
test_result_path=None
max_epoch=20
lr=0.01
beta_min=0.9
beta_max=0.999
weight_decay=0.0
dropout_prob=0.25

no_cuda=False
gpu="cuda:0"
num_workers=4
env=None
print_sample_cycle=10
eval_method="exact"

find_hyperparams=False
num_trials=100
angular_margin_loss=False
angular_margin=0.5
inverse_temp=30.0
infer_method_name=True
infer_variable_name=False
shuffle_variable_indexes=False

device = torch.device(gpu if not no_cuda and torch.cuda.is_available() else "cpu")
logger.info("device: {0}".format(device))


class Option(object):
    """configurations of the model"""

    def __init__(self, reader):
        self.max_path_length = max_path_length
        self.fcount = 8
        self.terminal_count = 24171
        self.path_count = 247461
        self.label_count = 9

        self.terminal_embed_size = terminal_embed_size
        self.path_embed_size = path_embed_size
        self.encode_size = encode_size

        self.dropout_prob = dropout_prob
        self.batch_size = batch_size
        self.eval_method = eval_method

        self.angular_margin_loss = angular_margin_loss
        self.angular_margin = angular_margin
        self.inverse_temp = inverse_temp

        self.device = device


def train():
    """train the model"""
    torch.manual_seed(random_seed)

    reader = DatasetReader(corpus_path)
    option = Option(reader)

    builder = DatasetBuilder(reader, option)

    # label_freq = torch.tensor(, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    model = Code2Vec(option).to(device)
    # print(model)
    # for param in model.parameters():
    #     print(type(param.data), param.size())

    learning_rate = lr
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(beta_min, beta_max), weight_decay=weight_decay)

    _train(model, optimizer, criterion, option, reader, builder, None)
    accuracy = _test(model,option,reader,builder)

def _test(model,option,reader,builder):

    model.eval()
    # with torch.no_grad():
    expected_labels = []
    actual_labels = []
    test_dataset = builder.refresh_test_dataset
    starts = test_dataset.starts
    paths = test_dataset.paths
    ends = test_dataset.ends
    label = test_dataset.labels
    expected_labels.extend(label)

    preds, _, _ = model.forward(starts, paths, ends, label)
        # loss = calculate_loss(preds, label, criterion, option)
        # test_loss += loss.item()
    _, preds_label = torch.max(preds, dim=1)
    actual_labels.extend(preds_label)

    expected_labels = np.array(expected_labels)
    actual_labels = np.array(actual_labels)
    accuracy, precision, recall, f1 = None, None, None, None
    if eval_method == 'exact':
        accuracy, precision, recall, f1 = exact_match(expected_labels, actual_labels)
    return accuracy

def _train(model, optimizer, criterion, option, reader, builder, trial):
    """train the model"""

    f1 = 0.0
    best_f1 = None
    last_loss = None
    last_accuracy = None
    bad_count = 0

    if env == "tensorboard":
        summary_writer = SummaryWriter()
    else:
        summary_writer = None

    try:
        for epoch in range(max_epoch):
            train_loss = 0.0

            builder.refresh_train_dataset()
            train_data_loader = batch_data(builder.train_dataset, option.batch_size,option.fcount)

            model.train()
            for i_batch, sample_batched in enumerate(train_data_loader):
                starts = sample_batched.starts
                paths = sample_batched.paths
                ends = sample_batched.ends
                label = sample_batched.labels
                # print(len(starts))
                # print(len(label))
                optimizer.zero_grad()
                preds, _, _ = model.forward(starts, paths, ends, label)
                loss = calculate_loss(preds, label, criterion, option)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            builder.refresh_val_dataset()
            val_data_loader = batch_data(builder.val_dataset, option.batch_size,option.fcount)
            val_loss, accuracy, precision, recall, f1 = test(model, val_data_loader, criterion, option)

            if env == "floyd":
                print("epoch {0}".format(epoch))
                print('{{"metric": "train_loss", "value": {0}}}'.format(train_loss))
                print('{{"metric": "val_loss", "value": {0}}}'.format(val_loss))
                print('{{"metric": "accuracy", "value": {0}}}'.format(accuracy))
                # print('{{"metric": "precision", "value": {0}}}'.format(precision))
                # print('{{"metric": "recall", "value": {0}}}'.format(recall))
                # print('{{"metric": "f1", "value": {0}}}'.format(f1))
            else:
                logger.info("epoch {0}".format(epoch))
                logger.info('{{"metric": "train_loss", "value": {0}}}'.format(train_loss))
                logger.info('{{"metric": "val_loss", "value": {0}}}'.format(val_loss))
                logger.info('{{"metric": "accuracy", "value": {0}}}'.format(accuracy))
                # logger.info('{{"metric": "precision", "value": {0}}}'.format(precision))
                # logger.info('{{"metric": "recall", "value": {0}}}'.format(recall))
                # logger.info('{{"metric": "f1", "value": {0}}}'.format(f1))
            if env == "tensorboard":
                summary_writer.add_scalar('metric/train_loss', train_loss, epoch)
                summary_writer.add_scalar('metric/val_loss', val_loss, epoch)
                summary_writer.add_scalar('metric/accuracy', accuracy, epoch)
                # summary_writer.add_scalar('metric/precision', precision, epoch)
                # summary_writer.add_scalar('metric/recall', recall, epoch)
                # summary_writer.add_scalar('metric/f1', f1, epoch)

            if trial is not None:
                intermediate_value = 1.0 - f1
                trial.report(intermediate_value, epoch)
                if trial.should_prune(epoch):
                    raise optuna.structs.TrialPruned()

            # if epoch > 1 and epoch % print_sample_cycle == 0 and trial is None:
            #     print_sample(reader, model, test_data_loader, option)

            if best_f1 is None or best_f1 < f1:
                if env == "floyd":
                    print('{{"metric": "best_f1", "value": {0}}}'.format(f1))
                else:
                    logger.info('{{"metric": "best_f1", "value": {0}}}'.format(f1))
                if env == "tensorboard":
                    summary_writer.add_scalar('metric/best_f1', f1, epoch)

                best_f1 = f1
                if trial is None:
                    # vector_file = vectors_path
                    # with open(vector_file, "w") as f:
                    #     f.write("{0}\t{1}\n".format(len(reader.items), option.encode_size))
                    # write_code_vectors(reader, model, train_data_loader, option, vector_file, "a", None)
                    # write_code_vectors(reader, model, test_data_loader, option, vector_file, "a", test_result_path)
                    torch.save(model.state_dict(), path.join(model_path, "code2vec.model"))

            if last_loss is None or train_loss < last_loss or last_accuracy is None or last_accuracy < accuracy:
                last_loss = train_loss
                last_accuracy = accuracy
                bad_count = 0
            else:
                bad_count += 1
            if bad_count > 10:
                print('early stop loss:{0}, bad:{1}'.format(train_loss, bad_count))
                print_sample(reader, model, test_data_loader, option)
                break

    finally:
        if env == "tensorboard":
            summary_writer.close()

    return 1.0 - f1


def calculate_loss(predictions, label, criterion, option):
    # preds = F.log_softmax(predictions, dim=1)
    #
    # batch_size = predictions.size()[0]
    # y_onehot = torch.FloatTensor(batch_size, option.label_count).to(device)
    # y_onehot.zero_()
    # y_onehot.scatter_(1, label.view(-1, 1), 1)
    #
    # loss = -torch.mean(torch.sum(preds * y_onehot, dim=1))

    preds = F.log_softmax(predictions, dim=1)
    loss = criterion(preds, label)

    return loss


def test(model, data_loader, criterion, option):
    """test the model"""
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        expected_labels = []
        actual_labels = []

        for i_batch, sample_batched in enumerate(data_loader):
            starts = sample_batched.starts
            paths = sample_batched.paths
            ends = sample_batched.ends
            label = sample_batched.labels
            # print(label)
            expected_labels.extend(label)

            preds, _, _ = model.forward(starts, paths, ends, label)
            loss = calculate_loss(preds, label, criterion, option)
            test_loss += loss.item()
            _, preds_label = torch.max(preds, dim=1)
            actual_labels.extend(preds_label)

        expected_labels = np.array(expected_labels)

        actual_labels = np.array(actual_labels)
        accuracy, precision, recall, f1 = None, None, None, None
        if eval_method == 'exact':
            accuracy, precision, recall, f1 = exact_match(expected_labels, actual_labels)
        # elif eval_method == 'subtoken':
        #     accuracy, precision, recall, f1 = subtoken_match(expected_labels, actual_labels, label_vocab)
        # elif eval_method == 'ave_subtoken':
        #     accuracy, precision, recall, f1 = averaged_subtoken_match(expected_labels, actual_labels, label_vocab)
        return test_loss, accuracy, precision, recall, f1


def exact_match(expected_labels, actual_labels):
    expected_labels = np.array(expected_labels, dtype=np.uint64)
    actual_labels = np.array(actual_labels, dtype=np.uint64)
    precision, recall, f1, _ = precision_recall_fscore_support(expected_labels, actual_labels, average='weighted')
    accuracy = accuracy_score(expected_labels, actual_labels)
    return accuracy, precision, recall, f1


def print_sample(reader, model, data_loader, option):
    """print one data that leads correct prediction with the trained model"""
    model.eval()
    with torch.no_grad():
        for i_batch, sample_batched in enumerate(data_loader):
            starts = sample_batched['starts'].to(option.device)
            paths = sample_batched['paths'].to(option.device)
            ends = sample_batched['ends'].to(option.device)
            label = sample_batched['label'].to(option.device)

            preds, code_vector, attn = model.forward(starts, paths, ends, label)
            _, preds_label = torch.max(preds, dim=1)

            for i in range(len(starts)):
                if preds_label[i] == label[i]:
                    # 予測と正解が一致したデータを1つだけ表示する。
                    start_names = [reader.terminal_vocab.itos[v.item()] for v in starts[i]]
                    path_names = [reader.path_vocab.itos[v.item()] for v in paths[i]]
                    end_names = [reader.terminal_vocab.itos[v.item()] for v in ends[i]]
                    label_name = reader.label_vocab.itos[label[i].item()]
                    pred_label_name = reader.label_vocab.itos[preds_label[i].item()]
                    attentions = attn.cpu()[i]

                    for start, path, end, attention in zip(start_names, path_names, end_names, attentions):
                        if start != "<PAD/>":
                            logger.info("{0} {1} {2} [{3}]".format(start, path, end, attention))
                    logger.info('expected label: {0}'.format(label_name))
                    logger.info('actual label:   {0}'.format(pred_label_name))
                    return


def write_code_vectors(reader, model, data_loader, option, vector_file, mode, test_result_file):
    """sav the code vectors"""
    model.eval()
    with torch.no_grad():
        if test_result_file is not None:
            fr = open(test_result_file, "w")
        else:
            fr = None

        with open(vector_file, mode) as fv:
            for i_batch, sample_batched in enumerate(data_loader):
                id = sample_batched['id']
                starts = sample_batched.starts.to(option.device)
                paths = sample_batched['paths'].to(option.device)
                ends = sample_batched['ends'].to(option.device)
                label = sample_batched['label'].to(option.device)

                preds, code_vector, _ = model.forward(starts, paths, ends, label)
                preds_prob, preds_label = torch.max(preds, dim=1)

                for i in range(len(starts)):
                    label_name = reader.label_vocab.itos[label[i].item()]
                    vec = code_vector.cpu()[i]
                    fv.write(label_name + "\t" + " ".join([str(e.item()) for e in vec]) + "\n")

                    if test_result_file is not None:
                        pred_name = reader.label_vocab.itos[preds_label[i].item()]
                        fr.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(id[i].item(), label_name == pred_name, label_name, pred_name, preds_prob[i].item()))

        if test_result_file is not None:
            fr.close()

def get_optimizer(trial, model):
    # optimizer = trial.suggest_categorical('optimizer', [adam, momentum])
    # weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
    # return optimizer(model, trial, weight_decay)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
    return adam(model, trial, weight_decay)


def adam(model, trial, weight_decay):
    lr = trial.suggest_loguniform('adam_lr', 1e-5, 1e-1)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def momentum(model, trial, weight_decay):
    lr = trial.suggest_loguniform('momentum_sgd_lr', 1e-5, 1e-1)
    return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)


#
# entry point
#
def main():
    if find_hyperparams:
        find_optimal_hyperparams()
    else:
        train()


if __name__ == '__main__':
    main()
