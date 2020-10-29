import argparse

from Config.config import Config
from DataProcessing.data_read import *
from Vocab.vocab import *
from Train.train import train
from Model.BertModel import MyBertModel
from transformers import BertTokenizer

if __name__ == '__main__':

    print("Process ID {}, Process Parent ID {}".format(os.getpid(), os.getppid()))
    tokenizer = BertTokenizer.from_pretrained('RoBERTa_zh_Large_PyTorch/.')

    # seed
    random_seed(520)

    # gpu
    gpu = torch.cuda.is_available()
    if gpu:
        print('The train will be using GPU.')
    else:
        print('The train will be using CPU.')
    print('CuDNN', torch.backends.cudnn.enabled)

    # config
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--config_file', type=str, default='./Config/config.ini')
    args = arg_parser.parse_args()
    config = Config(args.config_file)
    if gpu:
        config.add_args('Train', 'use_cuda', 'True')

    word_data_loader = ReadData(config)

    kd_tra_data_set, ca_tra_data_set, kd_dev_data_set, ca_dev_data_set = word_data_loader.read_data(tokenizer)
    kd_tra_data, kd_tra_data_gold = kd_tra_data_set[0], kd_tra_data_set[1]
    ca_tra_data, ca_tra_data_gold = ca_tra_data_set[0], ca_tra_data_set[1]
    kd_dev_data, kd_dev_data_gold = kd_dev_data_set[0], kd_dev_data_set[1]
    ca_dev_data, ca_dev_data_gold = ca_dev_data_set[0], ca_dev_data_set[1]

    # vocab
    if os.path.isfile(config.fact_word_src_vocab):
        tag_vocab = read_pkl(config.fact_word_tag_vocab)
    else:
        if not os.path.isdir(config.save_vocab_path):
            os.makedirs(config.save_vocab_path)
        tag_vocab = FactWordTagVocab([kd_tra_data, ca_tra_data])
        pickle.dump(tag_vocab, open(config.fact_word_tag_vocab, 'wb'))

    bert_model = MyBertModel(config)

    if config.use_cuda:
        bert_model = bert_model.cuda()

    # train
    train(bert_model, kd_tra_data, ca_tra_data, kd_dev_data, ca_dev_data, tag_vocab, config, kd_tra_data_gold, ca_tra_data_gold,
          kd_dev_data_gold, ca_dev_data_gold, tokenizer)
