import numpy as np
import os
import time

from transformers import AdamW, get_linear_schedule_with_warmup
from DataProcessing.data_batchiter import create_tra_batch
from Model.loss import *


def train(bert_model, kd_tra_data, ca_tra_data, kd_dev_data, ca_dev_data, tag_vocab, config, kd_tra_data_gold,
          ca_tra_data_gold, kd_dev_data_gold, ca_dev_data_gold, tokenizer):

    kd_tra_data.extend(ca_tra_data)
    pre_len = len(kd_tra_data_gold) + len(ca_tra_data_gold)
    kd_tra_data_gold.update(ca_tra_data_gold)
    if pre_len != len(kd_tra_data_gold):
        print('Error!!!, ID is not unique!!!')

    batch_num = int(np.ceil(len(kd_tra_data) / float(config.tra_batch_size)))

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay},
        {'params': [p for n, p in bert_model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer_bert = AdamW(optimizer_grouped_parameters, lr=config.bert_lr, eps=config.epsilon)
    scheduler_bert = get_linear_schedule_with_warmup(optimizer_bert, num_warmup_steps=0, num_training_steps=config.epoch * batch_num)

    # Get start!
    global_step = 0

    best_tra = 0
    best_dev_kd = 0
    best_dev_ca = 0

    for epoch in range(0, config.epoch):
        qa_ids = []
        score = 0
        print('\nThe epoch is starting.')
        epoch_start_time = time.time()
        batch_iter = 0
        print('The epoch is :', str(epoch))
        for word_batch in create_tra_batch(kd_tra_data, tag_vocab, config.tra_batch_size, config, tokenizer, shuffle=True):
            start_time = time.time()
            bert_model.train()
            batch_size = word_batch[0][0].size(0)
            p_h_tensor = word_batch[0]
            p_mask = word_batch[1]
            h_mask = word_batch[2]
            target = word_batch[3]
            qa_id = word_batch[4]

            logits = bert_model(p_h_tensor)
            loss, correct, accuracy, new_qa_ids = qa_predict(logits, target, qa_id)
            qa_ids.extend(new_qa_ids)
            loss = loss / config.update_every
            loss.backward()
            loss_value = loss.item()
            during_time = float(time.time() - start_time)
            print('Step:{}, Epoch:{}, batch_iter:{}, accuracy:{:.4f}({}/{}),'
                  'time:{:.2f}, loss:{:.6f}'.format(global_step, epoch, batch_iter, accuracy, correct, batch_size,
                                                    during_time, loss_value))
            batch_iter += 1

            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                if config.clip_max_norm_use:
                    nn.utils.clip_grad_norm_(bert_model.parameters(), max_norm=config.clip)
                if config.use_lr_decay:
                    scheduler_bert.step()
                optimizer_bert.step()
                bert_model.zero_grad()
                global_step += 1
            score += correct

            if batch_iter % config.test_interval == 0 or batch_iter == batch_num:
                print("now bert lr is {}".format(optimizer_bert.param_groups[0].get("lr")), '\n')
                dev_kd_score = evaluate(bert_model, kd_dev_data, kd_dev_data_gold, config, tag_vocab, tokenizer)
                if best_dev_kd < dev_kd_score:
                    print('the best kd_dev score is: acc:{}'.format(dev_kd_score))
                    best_dev_kd = dev_kd_score

                dev_ca_score = evaluate(bert_model, ca_dev_data, ca_dev_data_gold, config, tag_vocab, tokenizer, test=True)
                if best_dev_ca < dev_ca_score:
                    print('the best ca_dev score is: acc:{}'.format(dev_ca_score) + '\n')
                    best_dev_ca = dev_ca_score
                    if os.path.exists(config.save_model_path):
                        torch.save(bert_model.state_dict(), config.bert_model_pkl)
                    else:
                        os.makedirs(config.save_model_path)
                        torch.save(bert_model.state_dict(), config.bert_model_pkl)
        epoch_time = float(time.time() - epoch_start_time)
        tra_score = run_score(qa_ids, kd_tra_data_gold, tag_vocab)
        if tra_score > best_tra:
            best_tra = tra_score
            print('the best_train score is: acc:{}'.format(tra_score))
        print("epoch_time is:", epoch_time)


def evaluate(bert_model, dev_data, gold, config, tag_vocab, tokenizer, test=False):
    bert_model.eval()
    get_score = 0
    start_time = time.time()
    qa_ids_eval = []
    for word_batch in create_tra_batch(dev_data, tag_vocab, config.test_batch_size, config, tokenizer):
        batch_size = word_batch[0][0].size(0)
        p_h_tensor = word_batch[0]
        p_mask = word_batch[1]
        h_mask = word_batch[2]
        target = word_batch[3]
        qa_id_eval = word_batch[4]
        logits = bert_model(p_h_tensor)
        loss, correct, accuracy, new_qa_ids_eval = qa_predict(logits, target, qa_id_eval)
        qa_ids_eval.extend(new_qa_ids_eval)
        get_score += correct

    if test:
        dev_score = run_score(qa_ids_eval, gold, tag_vocab)
        print('the current_test ca_score is: acc:{}'.format(dev_score))
    else:
        dev_score = run_score(qa_ids_eval, gold, tag_vocab)
        print('the current_dev kd_score is: acc:{}'.format(dev_score))
    during_time = float(time.time() - start_time)
    print('spent time is:{:.4f}'.format(during_time))
    return dev_score


def run_score(qa_ids, data_gold, tag_vocab):
    count = 0
    value_dict = {}
    pre_value = []
    value_sorts = sorted(qa_ids, key=lambda k: (k[0], k[1]), reverse=False)
    for i in value_sorts:
        pre_value.append(tag_vocab.id2word(i[2]))
        if len(pre_value) == 4:
            value_dict[i[0]] = pre_value
            pre_value = []

    for key in value_dict:
        if value_dict[key] == data_gold[key]:
            count += 1
    acc = count / len(data_gold)
    return acc



