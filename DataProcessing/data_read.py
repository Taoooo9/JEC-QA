import os
import jieba

from DataProcessing.Dependency import readDepTree
from Units.units import *


class JEC_Data(object):

    def __init__(self, sentence_ID, number_ID, gold_label, sentence1, sentence12parser, sentence1_list,
                 sentence2, sentence22parser, sentence2_list, subject, type):
        self.sentence_ID = sentence_ID
        self.number_ID = number_ID
        self.gold_label = gold_label
        self.sentence1 = sentence1
        self.sentence12parser = sentence12parser
        self.sentence1_list = sentence1_list
        self.sentence2 = sentence2
        self.sentence22parser = sentence22parser
        self.sentence2_list = sentence2_list
        self.subject = subject
        self.type = type


class ReadData(object):

    def __init__(self, config):
        self.config = config
        if not os.path.isdir(config.save_dir):
            os.makedirs(config.save_dir)
        if not os.path.isdir(config.save_pkl_path):
            os.makedirs(config.save_pkl_path)

    def read_data(self, tokenizer):
        if os.path.isfile(self.config.kd_train_data_word_pkl):
            kd_tra_data = read_pkl(self.config.kd_train_data_word_pkl)
            ca_tra_data = read_pkl(self.config.ca_train_data_word_pkl)
        else:
            kd_tra_data = self.load_data(self.config.kd_train_file, tokenizer)
            ca_tra_data = self.load_data(self.config.ca_train_file, tokenizer)
            pickle.dump(kd_tra_data, open(self.config.kd_train_data_word_pkl, 'wb'))
            pickle.dump(ca_tra_data, open(self.config.ca_train_data_word_pkl, 'wb'))
        if os.path.isfile(self.config.kd_dev_data_word_pkl):
            kd_dev_data = read_pkl(self.config.kd_dev_data_word_pkl)
            ca_dev_data = read_pkl(self.config.ca_dev_data_word_pkl)
        else:
            kd_dev_data = self.load_data(self.config.kd_dev_file, tokenizer)
            ca_dev_data = self.load_data(self.config.ca_dev_file, tokenizer)
            pickle.dump(kd_dev_data, open(self.config.kd_dev_data_word_pkl, 'wb'))
            pickle.dump(ca_dev_data, open(self.config.ca_dev_data_word_pkl, 'wb'))
        return kd_tra_data, ca_tra_data, kd_dev_data, ca_dev_data

    def load_data(self, file, tokenizer):
        data = []
        gold_label = {}
        sentence12parser = ''
        sentence22parser = ''
        with open(file, encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                gold_unit = []
                line = eval(line)
                sentence1 = line['statement']
                if line['answer']:
                    for value in ['A', 'B', 'C', 'D']:
                        sentence2 = line['option_list'][value]
                        bert_piece1 = tokenizer.encode_plus(text=sentence1, add_special_tokens=True, return_tensors='pt')
                        bert_piece2 = tokenizer.encode_plus(text=sentence2, add_special_tokens=True, return_tensors='pt')
                        bert_len = bert_piece1["input_ids"].size()[1] + bert_piece2["input_ids"].size()[1]
                        while bert_len > self.config.max_length:
                            sentence1 = sentence1[bert_len - self.config.max_length + 1:]
                            bert_piece = tokenizer.encode_plus(text=sentence1, text_pair=sentence2,
                                                               add_special_tokens=True, return_tensors='pt')
                            bert_len = bert_piece["input_ids"].size()[1]

                sentence1_list = jieba.lcut(sentence1)
                if line['answer']:
                    for value in ['A', 'B', 'C', 'D']:
                        if value in line['answer']:
                            gold_unit.append('Yes')
                            sentence2 = line['option_list'][value]
                            sentence2_list = jieba.lcut(sentence2)
                            if 'subject' in line.keys():
                                data.append(
                                    JEC_Data(line['id'], value, 'Yes', sentence1, sentence12parser, sentence1_list,
                                             sentence2,
                                             sentence22parser, sentence2_list, line['subject'], line['type']))
                            else:
                                data.append(
                                    JEC_Data(line['id'], value, 'Yes', sentence1, sentence12parser, sentence1_list,
                                             sentence2,
                                             sentence22parser, sentence2_list, None, line['type']))

                        else:
                            gold_unit.append('No')
                            sentence2 = line['option_list'][value]
                            sentence2_list = jieba.lcut(sentence2)
                            if 'subject' in line.keys():
                                data.append(
                                    JEC_Data(line['id'], value, 'No', sentence1, sentence12parser, sentence1_list,
                                             sentence2,
                                             sentence22parser, sentence2_list, line['subject'], line['type']))
                            else:
                                data.append(
                                    JEC_Data(line['id'], value, 'No', sentence1, sentence12parser, sentence1_list,
                                             sentence2,
                                             sentence22parser, sentence2_list, None, line['type']))
                if line['answer']:
                    gold_label[line['id']] = gold_unit
            return [data, gold_label]


def jec2parser(text, parser_vocab):
    split_data = jieba.lcut(text)
    data_set = []
    for idx, word in enumerate(split_data, 1):
        data_set.append([str(idx) + '\t' + word + '\t' + '_' + '\t' + '_' + '\t' + '_' + '\t' + '_' + '\t' + '_' + '\t' \
                         + '_' + '\t' + '_' + '\t' + '_'])
    sentence = readDepTree(data_set, parser_vocab)
    return sentence



