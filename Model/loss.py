import torch.nn as nn
import torch
import torch.nn.functional as F
import copy


def tri_loss(logit, config):
    p = config.p
    margin = config.margin
    count = 0
    batch_size = logit.size(0)
    logit = torch.split(logit, 1, 1)
    triplet_loss = nn.TripletMarginLoss(margin=margin, p=p)
    loss = triplet_loss(logit[0], logit[1], logit[2])
    anchor = logit[0]
    positive = logit[1]
    negative = logit[2]
    for i in range(batch_size):
        d1 = l_norm(p, anchor[i], positive[i])
        d2 = l_norm(p, anchor[i], negative[i])
        if d1 < d2:
            count += 1
    return loss, count


def l_norm(p, x, y):
    return torch.norm_except_dim(x - y, p)


def ln_activation_p(x, n, epsilon):
    return -torch.log(-torch.div(x, n) + 1 + epsilon)


def ln_activation_n(x, n, epsilon):
    return -torch.log(-torch.div(n - x, n) + 1 + epsilon)


def M_Loss(d1, d2, margin):
    return torch.max(d1 - d2 + margin, 0)


def class_loss(logit, gold):
    batch_size = logit.size(0)
    loss = F.cross_entropy(logit, gold)
    correct = (torch.max(logit, 1)[1].view(gold.size()).data == gold.data).sum()
    accuracy = 100.0 * correct / batch_size
    return loss, correct, accuracy


def qa_predict(logits, gold, qa_ids):
    new_ids = copy.deepcopy(qa_ids)
    batch_size = logits.size(0)
    loss = F.cross_entropy(logits, gold)
    predict_id = torch.max(logits, 1)[1].view(gold.size()).data
    correct = (torch.max(logits, 1)[1].view(gold.size()).data == gold.data).sum()
    accuracy = 100.0 * correct / batch_size
    for i in range(batch_size):
        new_ids[i].append(predict_id[i])
    return loss, correct, accuracy, new_ids


def less_loss(d1, d2, n, epsilon):
    pos_loss = ln_activation_p(d1, n, epsilon)
    neg_loss = ln_activation_n(d2, n, epsilon)
    loss = pos_loss + neg_loss
    return loss


def less_triplet_loss(logit, config):
    p = config.p
    count = 0
    loss_sum = 0
    epsilon = config.epsilon
    n = len(logit.shape)
    batch_size = logit.size(0)
    for idx in range(batch_size):
        one_logit = logit[idx]
        one_logit = torch.split(one_logit, 1, 0)
        anchor = one_logit[0]
        positive = one_logit[1]
        negative = one_logit[2]
        d1 = l_norm(p, anchor, positive)
        d2 = l_norm(p, anchor, negative)
        loss = less_loss(d1, d2, n, epsilon)
        loss_sum += loss
        if d1 < d2:
            count += 1
    avg_loss = loss_sum / batch_size
    return avg_loss, count





