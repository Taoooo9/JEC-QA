from configparser import ConfigParser


class Config(object):

    def __init__(self, config_file):
        config = ConfigParser()
        config.read(config_file)
        for section in config.sections():
            for k, v in config.items(section):
                print(k, ":", v)
        self._config = config
        self.config_file = config_file
        config.write(open(config_file, 'w+'))

    def add_args(self, section, key, value):
        if self._config.has_section(section):
            print('This is a section already.')
        else:
            print('Now, we will add a new section.')
            self._config.add_section(section)
        if self._config.has_option(section, key):
            self._config.set(section, key, value)
            print('Add parameter successfully.')
        self._config.write(open(self.config_file, 'w'))

    # Dataset
    @property
    def kd_train_file(self):
        return self._config.get('Dataset', 'kd_train_file')

    @property
    def ca_train_file(self):
        return self._config.get('Dataset', 'ca_train_file')

    @property
    def kd_dev_file(self):
        return self._config.get('Dataset', 'kd_dev_file')

    @property
    def ca_dev_file(self):
        return self._config.get('Dataset', 'ca_dev_file')

    @property
    def kd_train_file_p(self):
        return self._config.get('Dataset', 'kd_train_file_p')

    @property
    def kd_train_file_h(self):
        return self._config.get('Dataset', 'kd_train_file_h')

    @property
    def ca_train_file_p(self):
        return self._config.get('Dataset', 'ca_train_file_p')

    @property
    def ca_train_file_h(self):
        return self._config.get('Dataset', 'ca_train_file_h')

    @property
    def kd_dev_file_p(self):
        return self._config.get('Dataset', 'kd_dev_file_p')

    @property
    def kd_dev_file_h(self):
        return self._config.get('Dataset', 'kd_dev_file_h')

    @property
    def ca_dev_file_p(self):
        return self._config.get('Dataset', 'ca_dev_file_p')

    @property
    def ca_dev_file_h(self):
        return self._config.get('Dataset', 'ca_dev_file_h')

    @property
    def embedding_file(self):
        return self._config.get('Dataset', 'embedding_file')

    # Save
    @property
    def save_dir(self):
        return self._config.get('Save', 'save_dir')

    @property
    def save_pkl_path(self):
        return self._config.get('Save', 'save_pkl_path')

    @property
    def save_model_path(self):
        return self._config.get('Save', 'save_model_path')

    @property
    def save_vocab_path(self):
        return self._config.get('Save', 'save_vocab_path')

    @property
    def bert_model_pkl(self):
        return self._config.get('Save', 'bert_model_pkl')

    @property
    def nli_model_pkl(self):
        return self._config.get('Save', 'nli_model_pkl')

    @property
    def kd_train_data_word_pkl(self):
        return self._config.get('Save', 'kd_train_data_word_pkl')

    @property
    def ca_train_data_word_pkl(self):
        return self._config.get('Save', 'ca_train_data_word_pkl')

    @property
    def kd_dev_data_word_pkl(self):
        return self._config.get('Save', 'kd_dev_data_word_pkl')

    @property
    def ca_dev_data_word_pkl(self):
        return self._config.get('Save', 'ca_dev_data_word_pkl')

    @property
    def embedding_pkl(self):
        return self._config.get('Save', 'embedding_pkl')

    @property
    def train_word_data_iter(self):
        return self._config.get('Save', 'train_word_data_iter')

    @property
    def dev_word_data_iter(self):
        return self._config.get('Save', 'dev_word_data_iter')

    @property
    def test_word_data_iter(self):
        return self._config.get('Save', 'test_word_data_iter')

    @property
    def fact_word_src_vocab(self):
        return self._config.get('Save', 'fact_word_src_vocab')

    @property
    def fact_word_tag_vocab(self):
        return self._config.get('Save', 'fact_word_tag_vocab')

    @property
    def load_dir(self):
        return self._config.get('Save', 'load_dir')

    @property
    def load_model_path(self):
        return self._config.get('Save', 'load_model_path')

    @property
    def load_vocab_path(self):
        return self._config.get('Save', 'load_vocab_path')

    # Train
    @property
    def use_cuda(self):
        return self._config.getboolean('Train', 'use_cuda')

    @property
    def epoch(self):
        return self._config.getint('Train', 'epoch')

    @property
    def tra_batch_size(self):
        return self._config.getint('Train', 'tra_batch_size')

    @property
    def test_batch_size(self):
        return self._config.getint('Train', 'test_batch_size')

    @property
    def use_lr_decay(self):
        return self._config.getboolean('Train', 'use_lr_decay')

    @property
    def clip_max_norm_use(self):
        return self._config.getboolean('Train', 'clip_max_norm_use')

    @property
    def test_interval(self):
        return self._config.getint('Train', 'test_interval')

    @property
    def early_stop(self):
        return self._config.getint('Train', 'early_stop')

    @property
    def update_every(self):
        return self._config.getint('Train', 'update_every')

    @property
    def loss(self):
        return self._config.get('Train', 'loss')

    @property
    def shuffle(self):
        return self._config.get('Train', 'shuffle')

    @property
    def scheduler_bert(self):
        return self._config.getboolean('Train', 'scheduler_bert')

    # Data_loader
    @property
    def data_cut(self):
        return self._config.getboolean('Data_loader', 'data_cut')

    @property
    def data_cut_k(self):
        return self._config.getint('Data_loader', 'data_cut_k')

    @property
    def stop_word(self):
        return self._config.getboolean('Data_loader', 'stop_word')

    @property
    def read_sen(self):
        return self._config.getboolean('Data_loader', 'read_sen')

    @property
    def read_char(self):
        return self._config.getboolean('Data_loader', 'read_char')

    # Model
    @property
    def embedding_word_dim(self):
        return self._config.getint('Model', 'embedding_word_dim')

    @property
    def embedding_word_num(self):
        return self._config.getint('Model', 'embedding_word_num')

    @property
    def max_length(self):
        return self._config.getint('Model', 'max_length')

    @property
    def srl_dim(self):
        return self._config.getint('Model', 'srl_dim')

    @property
    def parser_dim(self):
        return self._config.getint('Model', 'parser_dim')

    @property
    def input_channels(self):
        return self._config.getint('Model', 'input_channels')

    @property
    def primary_caps_output(self):
        return self._config.getint('Model', 'primary_caps_output')

    @property
    def hidden_size(self):
        return self._config.getint('Model', 'hidden_size')

    @property
    def bert_size(self):
        return self._config.getint('Model', 'bert_size')

    @property
    def caps_kernel_size(self):
        return self._config.get('Model', 'caps_kernel_size')

    @property
    def kernel_num(self):
        return self._config.getint('Model', 'kernel_num')

    @property
    def primary_caps_num(self):
        return self._config.getint('Model', 'primary_caps_num')

    @property
    def dropout(self):
        return self._config.getfloat('Model', 'dropout')

    @property
    def class_num(self):
        return self._config.getint('Model', 'class_num')

    @property
    def which_model(self):
        return self._config.get('Model', 'which_model')

    @property
    def learning_algorithm(self):
        return self._config.get('Model', 'learning_algorithm')

    @property
    def bert_lr(self):
        return self._config.getfloat('Model', 'bert_lr')

    @property
    def nli_lr(self):
        return self._config.getfloat('Model', 'nli_lr')

    @property
    def min_lr(self):
        return self._config.getfloat('Model', 'min_lr')

    @property
    def weight_decay(self):
        return self._config.getfloat('Model', 'weight_decay')

    @property
    def lr_rate_decay(self):
        return self._config.getfloat('Model', 'lr_rate_decay')

    @property
    def margin(self):
        return self._config.getfloat('Model', 'margin')

    @property
    def p(self):
        return self._config.getint('Model', 'p')

    @property
    def patience(self):
        return self._config.getint('Model', 'patience')

    @property
    def epsilon(self):
        return self._config.getfloat('Model', 'epsilon')

    @property
    def factor(self):
        return self._config.getfloat('Model', 'factor')

    @property
    def pre_embedding(self):
        return self._config.getboolean('Model', 'pre_embedding')

    @property
    def num_capsules(self):
        return self._config.getint('Model', 'num_capsules')

    @property
    def correct_bias(self):
        return self._config.getboolean('Model', 'correct_bias')

    @property
    def tune_start_layer(self):
        return self._config.getint('Model', 'tune_start_layer')

    @property
    def srl(self):
        return self._config.getboolean('Model', 'srl')

    @property
    def parser(self):
        return self._config.getboolean('Model', 'parser')

    @property
    def s_p(self):
        return self._config.getboolean('Model', 's_p')

    @property
    def sp_flag(self):
        return self._config.getboolean('Model', 'sp_flag')

    @property
    def decay(self):
        return self._config.getfloat('Model', 'decay')

    @property
    def decay_steps(self):
        return self._config.getfloat('Model', 'decay_steps')

    @property
    def clip(self):
        return self._config.getfloat('Model', 'clip')

    @property
    def beta_1(self):
        return self._config.getfloat('Model', 'beta_1')

    @property
    def beta_2(self):
        return self._config.getfloat('Model', 'beta_2')
