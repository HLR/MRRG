# coding=utf-8
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, L1Loss
from transformers import BertPreTrainedModel, RobertaModel, RobertaConfig

from modeling_encoder import TextEncoder, MODEL_NAME_TO_CLASS
from graph_utils.data_utils import *
from graph_utils.layers import *

logger = logging.getLogger(__name__)

# ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
#     'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
#     'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
# }


class RobertaForSequenceClassificationConsistency(BertPreTrainedModel):
    config_class = RobertaConfig
    # pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForSequenceClassificationConsistency, self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.classifier = RobertaClassificationHead(config) ## add graph rep

        self.class_loss_fct = CrossEntropyLoss()
        self.consistency_loss_fct = L1Loss()

        ### chen gate function
        self.gate = nn.Linear(768, 1)
        self.rel_gate = nn.Linear(768, 1)
        # self.classifier = nn.Linear(768 * 10, 3)
    
    # change the value of lambda
    def set_lambda(self, lambda_a, lambda_b):
        self.lambda_a = lambda_a
        self.lambda_b = lambda_b

    def forward(self, input_ids, attention_mask, token_type_ids,
                adj, X, start_attn, end_attn, uni_attn, trans_attn, 
                position_ids=None, head_mask=None, labels=None,
                labels_one_hot=None, aug_labels_one_hot=None, paired=False, triplet=False, top_k=10, ):

        outputs = self.roberta(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=None,
                               position_ids=position_ids,
                               head_mask=head_mask)

        # pred for original data as usual
        sequence_output = outputs[0]

        ### chen code begin
        ### start gate function for q and doc
        # print(sequence_output.size())
        gate_format = sequence_output.view(-1, sequence_output.size()[2]) ## (bz * max_len, 768)
        gate_score = self.gate(gate_format) ## (bz * max_len, 1)
        sequence_output_gate = gate_score.view(sequence_output.size()[0], sequence_output.size()[1]) #### (bz, max_len)
        sequence_output_gate_softmax = F.softmax(sequence_output_gate, dim=1)
        ### choose top k
        top_k_score, topk_k_index = torch.topk(sequence_output_gate_softmax, top_k, sorted=False) ##(bz, topk)   
        ### retrieve the top k representations from roberta
        relevant_word_emb = []
        for i in range(0, sequence_output.size()[0]):
            tmp = torch.index_select(sequence_output[i], 0, topk_k_index[i]) ## 
            relevant_word_emb.append(tmp)
        lang_relat = torch.cat(relevant_word_emb, 0) ## [bz * topk, 768]
        lang_candidate_relations = lang_relat.view(sequence_output.size()[0], top_k, sequence_output.size()[2])
        lang_relat = lang_relat.view(sequence_output.size()[0], top_k * sequence_output.size()[2]) ## [bz, topk * 768] it should change in the later time.

        ## relation topk * (topk - 1) // 2
        lang_candidate_relations_1 = lang_relat.view(sequence_output.size()[0], top_k, 1,  sequence_output.size()[2])
        lang_candidate_relations_2 = lang_relat.view(sequence_output.size()[0], 1, top_k,  sequence_output.size()[2])
        lang_candidate_relations_repeat_1 = lang_candidate_relations_1.repeat(1, 1, top_k, 1) 
        lang_candidate_relations_repeat_2 = lang_candidate_relations_2.repeat(1, top_k, 1, 1) 
        lang_candidate_relations = lang_candidate_relations_repeat_1 + lang_candidate_relations_repeat_2
        lang_candidate_relations = lang_candidate_relations.view(sequence_output.size()[0], top_k * top_k, sequence_output.size()[2])

        # import pdb
        # pdb.set_trace()
        relate_ind = torch.tril_indices(top_k, top_k, -1).cuda()
        # relate_ind = torch.tril_indices(lang_candidate_relations.size()[1], lang_candidate_relations.size()[1], -1).cuda()
        relate_ind[1] = relate_ind[1] * top_k
        relate_ind = relate_ind.sum(0)

        relate_stack = lang_candidate_relations.index_select(1, relate_ind)

        rel_gate_score = self.rel_gate(relate_stack) ## (bz * max_len, 1)
        relate_stack_rel_gate = rel_gate_score.view(relate_stack.size()[0], relate_stack.size()[1]) #### (bz, max_len)
        relate_stack_rel_gate_softmax = F.softmax(relate_stack_rel_gate, dim=1)
        ## choose top k
        top_k_rel_score, topk_k_rel_index = torch.topk(relate_stack_rel_gate_softmax, top_k, sorted=False) ##(bz, topk)   
        relate_emb = []
        for i in range(0, relate_stack.size()[0]):
            tmp = torch.index_select(relate_stack[i], 0, topk_k_rel_index[i]) 
            relate_emb.append(tmp)
        relation_rep = torch.cat(relate_emb, 0) ## [bz * topk, 768] 
        relation_rep = relation_rep.view(relate_stack.size()[0], top_k * relate_stack.size()[2]) ## [bz, topk * 768]

        lang_relat = torch.cat([lang_relat, relation_rep], 1) ## [bz, topk * 768 * 2]

        ### begin to classify
        # logits = self.classifier(lang_relat)  ### [bz, num_label]
        ## if add graph:
        logits = self.classifier(lang_relat, adj, X, start_attn, end_attn, uni_attn, trans_attn)  ### [bz, num_label]

        # outputs = (logits,) + outputs[2:]
        outputs =  F.softmax(logits, dim=1)
        
        class_loss = self.class_loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        ### chen code end

        loss = class_loss

        return (loss, outputs)  # (loss), (consistency_loss), (class_loss), logits, (hidden_states), (attentions)


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super(RobertaClassificationHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense = nn.Linear(768 * 10 * 2, config.hidden_size)
        # self.dense = nn.Linear(768 * 10, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.out_proj = nn.Linear(config.hidden_size, 3)

        ### chen graph add begin
        self.k = 2
        self.n_node = 50
        # self.n_head = 1
        self.n_head = 34
        self.h_size = 1024
        self.graph_hidden_size = 100
        self.diag_decompose=True
        self.graph_model = MultiHopMessagePassingLayerDeprecated(self.k, self.n_head, self.h_size, self.diag_decompose, 0)
        ### chen graph add end
        self.graph_dense = nn.Linear(self.h_size, self.graph_hidden_size)
        self.out_proj_graph = nn.Linear(config.hidden_size + self.graph_hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, 3)

    def forward(self, features, X, adj, start_attn, end_attn, uni_attn, trans_attn, **kwargs): ## 与MultiHopMessagePassingLayerDeprecated的forward参数保持一致
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        # x = self.dropout(x)
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x) ### torch.Size([bz, 768]) 

        graph_reps = self.graph_model(adj, X, start_attn, end_attn, uni_attn, trans_attn) ### (bz, num_nodes, emb_size) ### torch.Size([29808, 50, 1024])
        mean_graph_reps = torch.mean(graph_reps, 1) ### (bz, 1024)
        mean_graph_reps_mlp = self.graph_dense(mean_graph_reps) ### (bz, 100)

        x_and_graph = torch.cat([x, mean_graph_reps_mlp], 1) ## (bz, 868])
        x = self.out_proj_graph(x_and_graph) ## (bz, 768])
        x = self.out_proj(x) ## (bz, 3])
        return x



#################################################################################################
# graph code
#################################################################################################


class MultiHopMessagePassingLayerDeprecated(nn.Module):
    def __init__(self, k, n_head, hidden_size, diag_decompose, n_basis, eps=1e-20, ablation=[]):
        super().__init__()
        self.diag_decompose = diag_decompose
        self.k = k
        self.n_head = n_head
        self.n_basis = n_basis
        self.eps = eps
        self.ablation = ablation

        if diag_decompose:
            self.w_vs = nn.Parameter(torch.randn(k, hidden_size, n_head))
        elif n_basis == 0:
            self.w_vs = nn.Parameter(torch.randn(k, hidden_size, hidden_size * n_head))
        else:
            self.w_vs = nn.Parameter(torch.randn(k, hidden_size * hidden_size, n_basis))
            self.w_vs_co = nn.Parameter(torch.randn(k, n_basis, n_head))

    def forward(self, X, A, start_attn, end_attn, uni_attn, trans_attn):
        """
        X: tensor of shape (batch_size, n_node, h_size)
        A: tensor of shape (batch_size, n_head, n_node, n_node)
        start_attn: tensor of shape (batch_size, n_node)
        end_attn: tensor of shape (batch_size, n_node)
        uni_attn: tensor of shape (batch_size, n_head)
        trans_attn: tensor of shape (batch_size, n_head, n_head)
        ablation: list[str]
        """
        k, n_head = self.k, self.n_head
        bs, n_node, h_size = X.size()

        if self.diag_decompose:
            W = self.w_vs
        elif self.n_basis == 0:
            W = self.w_vs
        else:
            W = self.w_vs.bmm(self.w_vs_co).view(k, h_size, h_size * n_head)

        A = A.view(bs * n_head, n_node, n_node)
        uni_attn = uni_attn.view(bs * n_head)
        if 'no_trans' in self.ablation or 'no_att' in self.ablation:
            Z = X * start_attn.unsqueeze(2)
            for t in range(k):
                if self.diag_decompose:
                    Z = (Z.unsqueeze(-1) * W[t]).view(bs, n_node, h_size, n_head)
                else:
                    Z = Z.matmul(W[t]).view(bs, n_node, h_size, n_head)
                Z = Z.permute(0, 3, 1, 2).contiguous().view(bs * n_head, n_node, h_size)
                Z = Z * uni_attn[:, None, None]
                Z = A.bmm(Z)
                Z = Z.view(bs, n_head, n_node, h_size).sum(1)
            Z = Z * end_attn.unsqueeze(2)

            D = start_attn.clone()
            for t in range(k):
                D = D.repeat(1, n_head).view(bs * n_head, n_node, 1)
                D = D * uni_attn[:, None, None]
                D = A.bmm(D)
                D = D.view(bs, n_head, n_node).sum(1)
            D = D * end_attn

        else:
            Z = X * start_attn.unsqueeze(2)  # (bs, n_node, h_size)
            for t in range(k):
                if t == 0:
                    if self.diag_decompose:
                        Z = (Z.unsqueeze(-1) * W[t]).view(bs, n_node, h_size, n_head)
                    else:
                        Z = Z.matmul(W[t]).view(bs, n_node, h_size, n_head)
                else:
                    if self.diag_decompose:
                        Z = (Z.unsqueeze(-1) * W[t]).view(bs, n_head, n_node, h_size, n_head)
                    else:
                        Z = Z.matmul(W[t]).view(bs, n_head, n_node, h_size, n_head)
                    Z = Z * trans_attn[:, :, None, None, :]
                    Z = Z.sum(1)  # (bs, n_node, h_size,n_head)

                Z = Z.permute(0, 3, 1, 2).contiguous().view(bs * n_head, n_node, h_size)
                Z = Z * uni_attn[:, None, None]
                Z = A.bmm(Z)
                Z = Z.view(bs, n_head, n_node, h_size)
            if k >= 1:
                Z = Z.sum(1)
            Z = Z * end_attn.unsqueeze(2)

            # compute the normalization factor
            D = start_attn
            for t in range(k):
                if t == 0:
                    D = D.unsqueeze(1).expand(bs, n_head, n_node)
                else:
                    D = D.unsqueeze(2) * trans_attn.unsqueeze(3)
                    D = D.sum(1)
                D = D.contiguous().view(bs * n_head, n_node, 1)
                D = D * uni_attn[:, None, None]
                D = A.bmm(D)
                D = D.view(bs, n_head, n_node)
            if k >= 1:
                D = D.sum(1)
            D = D * end_attn  # (bs, n_node)

        Z = Z / (D.unsqueeze(2) + self.eps)
        return Z


class MultiHopMessagePassingLayer(nn.Module):
    def __init__(self, k, n_head, hidden_size, diag_decompose, n_basis, eps=1e-20, init_range=0.01, ablation=[]):
        super().__init__()
        self.diag_decompose = diag_decompose
        self.k = k
        self.n_head = n_head
        self.hidden_size = hidden_size
        self.n_basis = n_basis
        self.eps = eps
        self.ablation = ablation

        if diag_decompose and n_basis > 0:
            raise ValueError('diag_decompose and n_basis > 0 cannot be True at the same time')

        if diag_decompose:
            self.w_vs = nn.Parameter(torch.zeros(k, hidden_size, n_head + 1))  # the additional head is used for the self-loop
            self.w_vs.data.uniform_(-init_range, init_range)
        elif n_basis == 0:
            self.w_vs = nn.Parameter(torch.zeros(k, n_head + 1, hidden_size, hidden_size))
            self.w_vs.data.uniform_(-init_range, init_range)
        else:
            self.w_vs = nn.Parameter(torch.zeros(k, n_basis, hidden_size * hidden_size))
            self.w_vs.data.uniform_(-init_range, init_range)
            self.w_vs_co = nn.Parameter(torch.zeros(k, n_head + 1, n_basis))
            self.w_vs_co.data.uniform_(-init_range, init_range)

    def init_from_old(self, w_vs, w_vs_co=None):
        """
        w_vs: tensor of shape (k, h_size, n_head) or (k, h_size, h_size * n_head) or (k, h_size*h_size, n_basis)
        w_vs_co: tensor of shape (k, n_basis, n_head)
        """
        raise NotImplementedError()
        k, n_head, h_size = self.k, self.n_head, self.hidden_size
        if self.diag_decompose:
            self.w_vs.data.copy_(w_vs)
        elif self.n_basis == 0:
            self.w_vs.data.copy_(w_vs.view(k, h_size, h_size, n_head).permute(0, 3, 1, 2))
        else:
            self.w_vs.data.copy_(w_vs.permute(0, 2, 1))
            self.w_vs_co.data.copy_(w_vs_co.permute(0, 2, 1))

    def _get_weights(self):
        if self.diag_decompose:
            W, Wi = self.w_vs[:, :, :-1], self.w_vs[:, :, -1]
        elif self.n_basis == 0:
            W, Wi = self.w_vs[:, :-1, :, :], self.w_vs[:, -1, :, :]
        else:
            W = self.w_vs_co.bmm(self.w_vs).view(self.k, self.n_head, self.hidden_size, self.hidden_size)
            W, Wi = W[:, :-1, :, :], W[:, -1, :, :]

        k, h_size = self.k, self.hidden_size
        W_pad = [W.new_ones((h_size,)) if self.diag_decompose else torch.eye(h_size, device=W.device)]
        for t in range(k - 1):
            if self.diag_decompose:
                W_pad = [Wi[k - 1 - t] * W_pad[0]] + W_pad
            else:
                W_pad = [Wi[k - 1 - t].mm(W_pad[0])] + W_pad
        assert len(W_pad) == k
        return W, W_pad

    def decode(self, end_ids, ks, A, start_attn, uni_attn, trans_attn):
        """
        end_ids: tensor of shape (batch_size,)
        ks: tensor of shape (batch_size,)
        A: tensor of shape (batch_size, n_head, n_node, n_node)
        start_attn: tensor of shape (batch_size, n_node)
        end_attn: tensor of shape (batch_size, n_node)
        uni_attn: tensor of shape (batch_size, n_head)
        trans_attn: tensor of shape (batch_size, n_head, n_head)
        returns: list[tensor of shape (path_len,)]
        """
        bs, n_head, n_node, n_node = A.size()
        assert ((A == 0) | (A == 1)).all()

        path_ids = end_ids.new_zeros((bs, self.k * 2 + 1))
        path_lengths = end_ids.new_zeros((bs,))

        for idx in range(bs):
            back_trace = []
            end_id, k, adj = end_ids[idx], ks[idx], A[idx]
            uni_a, trans_a, start_a = uni_attn[idx], trans_attn[idx], start_attn[idx]

            if (adj[:, end_id, :] == 0).all():  # end_id is not connected to any other node
                path_ids[idx, 0] = end_id
                path_lengths[idx] = 1
                continue

            dp = F.one_hot(end_id, num_classes=n_node).float()  # (n_node,)
            assert 1 <= k <= self.k
            for t in range(k):
                if t == 0:
                    dp = dp.unsqueeze(0).expand(n_head, n_node)
                else:
                    dp = dp.unsqueeze(0) * trans_a.unsqueeze(-1)  # (n_head, n_head, n_node)
                    dp, ptr = dp.max(1)
                    back_trace.append(ptr)  # (n_head, n_node)
                dp = dp.unsqueeze(-1) * adj  # (n_head, n_node, n_node)
                dp, ptr = dp.max(1)
                back_trace.append(ptr)  # (n_head, n_node)
                dp = dp * uni_a.unsqueeze(-1)  # (n_head, n_node)
            dp, ptr = dp.max(0)
            back_trace.append(ptr)  # (n_node,)
            dp = dp * start_a
            dp, ptr = dp.max(0)
            back_trace.append(ptr)  # （)
            assert dp.dim() == 0
            assert len(back_trace) == k + (k - 1) + 2

            # re-construct path from back_trace
            path = end_id.new_zeros((2 * k + 1,))  # (k + 1) entities and k relations
            path[0] = back_trace.pop(-1)
            path[1] = back_trace.pop(-1)[path[0]]
            for p in range(2, 2 * k + 1):
                if p % 2 == 0:  # need to fill a entity id
                    path[p] = back_trace.pop(-1)[path[p - 1], path[p - 2]]
                else:  # need to fill a relation id
                    path[p] = back_trace.pop(-1)[path[p - 2], path[p - 1]]
            assert len(back_trace) == 0
            assert path[-1] == end_id
            path_ids[idx, :2 * k + 1] = path
            path_lengths[idx] = 2 * k + 1

        return path_ids, path_lengths

    def forward(self, X, A, start_attn, end_attn, uni_attn, trans_attn):
        """
        X: tensor of shape (batch_size, n_node, h_size)
        A: tensor of shape (batch_size, n_head, n_node, n_node)
        start_attn: tensor of shape (batch_size, n_node)
        end_attn: tensor of shape (batch_size, n_node)
        uni_attn: tensor of shape (batch_size, n_head)
        trans_attn: tensor of shape (batch_size, n_head, n_head)
        """
        k, n_head = self.k, self.n_head
        bs, n_node, h_size = X.size()

        W, W_pad = self._get_weights()  # (k, h_size, n_head) or (k, n_head, h_size h_size)

        A = A.view(bs * n_head, n_node, n_node)
        uni_attn = uni_attn.view(bs * n_head)

        Z_all = []
        Z = X * start_attn.unsqueeze(2)  # (bs, n_node, h_size)
        for t in range(k):
            if t == 0:  # Z.size() == (bs, n_node, h_size)
                Z = Z.unsqueeze(-1).expand(bs, n_node, h_size, n_head)
            else:  # Z.size() == (bs, n_head, n_node, h_size)
                Z = Z.permute(0, 2, 3, 1).view(bs, n_node * h_size, n_head)
                Z = Z.bmm(trans_attn).view(bs, n_node, h_size, n_head)
            if self.diag_decompose:
                Z = Z * W[t]  # (bs, n_node, h_size, n_head)
                Z = Z.permute(0, 3, 1, 2).contiguous().view(bs * n_head, n_node, h_size)
            else:
                Z = Z.permute(3, 0, 1, 2).view(n_head, bs * n_node, h_size)
                Z = Z.bmm(W[t]).view(n_head, bs, n_node, h_size)
                Z = Z.permute(1, 0, 2, 3).contiguous().view(bs * n_head, n_node, h_size)
            Z = Z * uni_attn[:, None, None]
            Z = A.bmm(Z)
            Z = Z.view(bs, n_head, n_node, h_size)
            Zt = Z.sum(1) * W_pad[t] if self.diag_decompose else Z.sum(1).matmul(W_pad[t])
            Zt = Zt * end_attn.unsqueeze(2)
            Z_all.append(Zt)

        # compute the normalization factor
        D_all = []
        D = start_attn
        for t in range(k):
            if t == 0:  # D.size() == (bs, n_node)
                D = D.unsqueeze(1).expand(bs, n_head, n_node)
            else:  # D.size() == (bs, n_head, n_node)
                D = D.permute(0, 2, 1).bmm(trans_attn)  # (bs, n_node, n_head)
                D = D.permute(0, 2, 1)
            D = D.contiguous().view(bs * n_head, n_node, 1)
            D = D * uni_attn[:, None, None]
            D = A.bmm(D)
            D = D.view(bs, n_head, n_node)
            Dt = D.sum(1) * end_attn
            D_all.append(Dt)

        Z_all = [Z / (D.unsqueeze(2) + self.eps) for Z, D in zip(Z_all, D_all)]
        assert len(Z_all) == k
        if 'agg_self_loop' in self.ablation:
            Z_all = [X] + Z_all
        return Z_all