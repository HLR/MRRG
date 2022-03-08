from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import json

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, RobertaConfig,
                          RobertaTokenizer, RobertaForSequenceClassification)
from model_chen_graph_gate import RobertaForSequenceClassificationConsistency
# from transformers import AdamW, WarmupLinearSchedule
from transformers import AdamW, get_linear_schedule_with_warmup

from wiqa_preprocess import multi_qa_output_modes as output_modes
from wiqa_preprocess import multi_qa_processors as processors
# from wiqa_preprocess import multi_qa_convert_examples_to_features, multi_qa_triplet_convert_examples_to_features, multi_qa_triplet_convert_examples_to_features_augmented_data
from wiqa_preprocess import multi_qa_convert_examples_to_features, multi_qa_triplet_convert_examples_to_features_augmented_data

from graph_utils.data_utils import *
from graph_utils.layers import *
from graph_utils.optimization_utils import OPTIMIZER_CLASSES
from graph_utils.parser_utils import *

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'roberta_cons':  (RobertaConfig, RobertaForSequenceClassificationConsistency, RobertaTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}

DECODER_DEFAULT_LR = {
    'wiqa': 1e-3,
}

def get_node_feature_encoder(encoder_name):
    return encoder_name.replace('-cased', '-uncased')


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def set_seed(args):
    # Following Tandon et al. (2019).
    random.seed(args.seed)
    np.random.seed(args.seed * 7)
    torch.manual_seed(args.seed * 23)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(
            "runs", args.tensorboard_output))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(
        train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (
            len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(
            train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = WarmupLinearSchedule(
    #     optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps = t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train models.
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d",
                args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d",
                args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs),
                            desc="Epoch", disable=args.local_rank not in [-1, 0])
    # Added here for reproductibility (even between python 2 and 3)
    set_seed(args)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            # load the paired examples
            batch = tuple(t.to(args.device) for t in batch)
            if args.use_consistency is True:
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': None,
                          # add consistent pairs / triples'
                          'aug_one_input_ids': batch[3],
                          'aug_one_attention_mask': batch[4],
                          'aug_one_token_type_ids': None,
                          'aug_two_input_ids': batch[6],
                          'aug_two_attention_mask': batch[7],
                          'aug_two_token_type_ids': None,
                          'labels': batch[9],
                          'labels_one_hot': batch[10],
                          'aug_labels_one_hot': batch[11],
                          'paired':         batch[12],
                          'triplet': batch[13],
                          'adj': batch[14],
                          'X': batch[14],
                          'start_attn': batch[15],
                          'end_attn': batch[16],
                          'uni_attn': batch[17],
                          'trans_attn': batch[18],
                          }
            else:
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': None,
                          'labels':         batch[3],
                          'adj': batch[4],
                          'X': batch[5],
                          'start_attn': batch[6],
                          'end_attn': batch[7],
                          'uni_attn': batch[8],
                          'trans_attn': batch[9],
                          }
            outputs = model(**inputs)
            # print(model)
            # outputs = model(input_ids=batch[0],attention_mask=batch[1], token_type_ids=None, labels=batch[3])
            if args.use_consistency is True:
                loss, tmp_train_trans_loss, tmp_train_sym_loss = outputs[0], outputs[1], outputs[2]
            else:
                loss = outputs[0]
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    # Only evaluate when single GPU otherwise metrics may not
                    # average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar(
                                'eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar(
                        'lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar(
                        'loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(
                        args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(
                        model, 'module') else model
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(
                        output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, cp_emb, prefix=""):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)
    is_eval = True if args.do_eval is True else False
    is_test = True if args.do_test is True else False

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset, example_ids = load_and_cache_examples(
            args, eval_task, tokenizer, cp_emb, evaluate=is_eval, test=is_test, return_example_id=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * \
            max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(
            eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Evaluation!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            # For evaluation, we only treat the model as single input information.
            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': None,
                          'labels': batch[3],
                          'adj': batch[4],
                          'X': batch[5],
                          'start_attn': batch[6],
                          'end_attn': batch[7],
                          'uni_attn': batch[8],
                          'trans_attn': batch[9],
                        }
                outputs = model(**inputs)
                if args.use_consistency is True:
                    tmp_eval_loss, tmp_eval_cons_loss, tmp_eval_class_loss, logits = outputs[
                        0], outputs[1], outputs[2], outputs[3]
                else:
                    tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        # save predicted probabilities, predicted label and ground_truth_answer
        output_result = {}
        for q_id, pred, out_label_id in zip(example_ids, preds, out_label_ids):
            output_result[q_id] = {"pred_prob": [float(prob) for prob in pred], "pred": int(
                np.argmax(pred)), "output_label": int(out_label_id)}

        output_pred_file = os.path.join(
            eval_output_dir, "prediction_results.json")
        with open(output_pred_file, 'w') as outfile:
            json.dump(output_result, outfile)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = simple_accuracy(preds, out_label_ids)
        results[eval_task] = result

    return results


def load_and_cache_examples(args, task, tokenizer, pretrained_concept_emb, 
                            evaluate=False, test=False,
                            return_example_id=False, use_contextualized=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the
        # dataset, and the others will use the cache
        torch.distributed.barrier()

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    if evaluate is True and test is False:
        cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
            'dev',
            list(filter(None, args.model_name_or_path.split('/'))).pop(),
            str(args.max_seq_length),
            str(task)))
    elif evaluate is False and test is True:
        cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
            'test',
            list(filter(None, args.model_name_or_path.split('/'))).pop(),
            str(args.max_seq_length),
            str(task)))
    elif evaluate is False and test is False:
        cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
            'train',
            list(filter(None, args.model_name_or_path.split('/'))).pop(),
            str(args.max_seq_length),
            str(task)))
    else:
        raise NotImplementedError(
            "the mode should be either of train, dev or test.")

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s",
                    cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]

        ## chen start
        # n_node, n_head, h_size = 10, 2, 100
        # n_node, n_head, h_size = 50, 1, 1024
        n_node, n_head, h_size = 50, 34, 1024 ## cannot change 34 and 1024. 34 is the number of the relations in conceptnet. 1024 is the pre-trained graph embedding size. 
                                              ## in the other word, n_head == n_rel. h_size = graph_emb_size
        ### chen end

        ### adj = *self.train_decoder_data, self.train_adj_data, n_rel = load_adj_data(train_adj_path, max_node_num, num_choice, emb_pk_path=train_embs_path if use_contextualized else None
        ### *decoder_data, adj, n_rel = load_adj_data(args.dev_adj, n_node, 1, emb_pk_path=args.dev_embs if use_contextualized else None)
        if evaluate is True and test is False:
            ### 200太大了，max_node改成50试试 ### concept_ids, node_type_ids, adj_lengths, adj_data,
            examples = processor.get_dev_examples(args.data_dir)
            concept_ids, node_type_ids, adj_lengths, adj_data, n_rel = load_adj_data(args.dev_adj, n_node, 1, emb_pk_path=args.dev_embs if use_contextualized else None)
            adj_empty = torch.zeros((len(examples), 1, n_rel - 1, n_node, n_node), dtype=torch.float32)
        elif evaluate is False and test is True:
            examples = processor.get_test_examples(args.data_dir)
            concept_ids, node_type_ids, adj_lengths, adj_data, n_rel = load_adj_data(args.test_adj, n_node, 1, emb_pk_path=args.test_embs if use_contextualized else None)
            adj_empty = torch.zeros((len(examples), 1, n_rel - 1, n_node, n_node), dtype=torch.float32)
        elif evaluate is False and test is False:
            examples = processor.get_train_examples(args.data_dir)
            concept_ids, node_type_ids, adj_lengths, adj_data, n_rel = load_adj_data(args.train_adj, n_node, 1, emb_pk_path=args.train_embs if use_contextualized else None)
            adj_empty = torch.zeros((len(examples), 1, n_rel - 1, n_node, n_node), dtype=torch.float32)
        else:
            raise NotImplementedError()

        # print(len(adj[0]))     ### 3 因为num_of_choice = 1
        # print(len(adj[0][0]))  ### 3
        # print(adj[0][0][0].size(), adj[0][0][1].size(), adj[0][0][2].size())  ### torch.Size([5030]) torch.Size([5030]) torch.Size([5030])
        # print(adj[1][0][0].size(), adj[1][0][1].size(), adj[1][0][2].size())  ### torch.Size([4398]) torch.Size([4398]) torch.Size([4398])
        # print(adj[2][0][0].size(), adj[2][0][1].size(), adj[2][0][2].size())  ### torch.Size([3792]) torch.Size([3792]) torch.Size([3792])

        # concept_ids, node_type_ids, adj_lengths
        # print(concept_ids.size()) # torch.Size([29808, 1, n_node])
        # print(node_type_ids.size()) # torch.Size([29808, 1, n_node])
        # print(adj_lengths.size()) # torch.Size([29808, 1])

        n_samples = len(examples)
        start_attn = torch.ones(n_samples, n_node)
        end_attn = torch.ones(n_samples, n_node)
        uni_attn = torch.ones(n_samples, n_head)
        trans_attn = torch.ones(n_samples, n_head * n_head)
        # start_attn, end_attn, uni_attn, trans_attn = [torch.exp(x - x.max(-1, keepdim=True)[0]) for x in (start_attn, end_attn, uni_attn, trans_attn)]
        uni_attn = uni_attn.view(n_samples, n_head)
        trans_attn = trans_attn.view(n_samples, n_head, n_head)

        adj_new = adj_empty
        adj_new[:] = 0
        for sample_id in range(n_samples):
            for choice_id, (i, j, k) in enumerate(adj_data[sample_id]): ## because number of choice = 1, so choice_id is also equal to 0
                adj_new[sample_id, choice_id, i, j, k] = 1

        # print(adj_new.size()) ### torch.Size([29808, 1, 34, 50, 50])
        adj = adj_new

        ### lookup ConceptNet conceptid embeddings from recorded conceptid
        load_pretrained_concept_emb = CustomizedEmbedding(concept_num=pretrained_concept_emb.size(0), concept_out_dim=h_size,
                                               use_contextualized=use_contextualized, concept_in_dim=h_size,
                                               pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=True)

        X = load_pretrained_concept_emb(concept_ids) # torch.Size([29808, 1, n_node, 1024]) 1024: pretrain_graph_embedding h_size, cannot change, unless you use the random initial embedding
        X = X.view(n_samples, n_node, h_size)

        features = multi_qa_convert_examples_to_features(examples,
                                                         adj, X,
                                                         start_attn, end_attn, uni_attn, trans_attn, 
                                                         tokenizer,
                                                         label_list=label_list,
                                                         max_length=args.max_seq_length,
                                                         output_mode=output_mode,
                                                         pad_on_left=False,
                                                         pad_token=tokenizer.convert_tokens_to_ids(
                                                             [tokenizer.pad_token])[0],
                                                         pad_token_segment_id=0,
                                                         )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s",
                        cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and (not evaluate or not test):
        # Make sure only the first process in distributed training process the
        # dataset, and the others will use the cache
        torch.distributed.barrier()

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long)
    ### graph chen begin
    ## adj, X, start_attn, end_attn, uni_attn, trans_attn,
    all_start_attn = torch.stack([f.start_attn for f in features])
    all_end_attn = torch.stack([f.end_attn for f in features])
    all_uni_attn = torch.stack([f.uni_attn for f in features])
    all_trans_attn = torch.stack([f.trans_attn for f in features])
    all_adj = torch.stack([f.adj for f in features])
    all_X = torch.stack([f.X for f in features])
    ### graph chen end

    if output_mode == "classification":
        all_labels = torch.tensor(
            [f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor(
            [f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_adj, all_X, all_start_attn, all_end_attn, all_uni_attn, all_trans_attn)
    if return_example_id is True:
        example_ids = [f.example_id for f in features]
        return dataset, example_ids
    return dataset


def main():
    # parser = argparse.ArgumentParser()
    parser = get_parser()
    args, _ = parser.parse_known_args()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model")
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    # Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the test et.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=12, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--num_train_epochs", default=5.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--lambda_val", default=0.5, type=float,
                        help="a weight term between two losses.")
    parser.add_argument("--lambda_a", default=0.5, type=float,
                        help="a weight term between two losses.")
    parser.add_argument("--lambda_b", default=0.5, type=float,
                        help="a weight term between two losses.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=500,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='',
                        help="For distant debugging.")
    parser.add_argument('--server_port', type=str,
                        default='', help="For distant debugging.")

    parser.add_argument('--tensorboard_output', type=str, default='',
                        help="Save tensorboard file under the runs/dir.")

    parser.add_argument('--random_sample', action='store_true',
                        help="randomly select augmented data")

    parser.add_argument('--use_consistency', action='store_true',
                        help="use Li et al. (2019) loss")
    parser.add_argument('--no_augmentation', action='store_true',
                        help="Set true if you do not use augmentation; otherwise you'll run over augmentation mode.")
    parser.add_argument("--random_ratio", default=1.0, type=float,
                        help="a random sampling ratio.")
    parser.add_argument("--num_choice", default=3, type=float,
                        help="A more B less C no effect")

    ###### graph args ######
    parser.add_argument('--mode', default='train', choices=['train', 'eval', 'pred', 'decode'], help='run training or evaluation')
    parser.add_argument('--save_dir', default=f'./saved_models/graph_gate/', help='model output directory')

    # graph data
    ### graph args are located in graph_utils/parser_utils.py

    # graph regularization
    parser.add_argument('--dropouti', type=float, default=0.1, help='dropout for embedding layer')
    parser.add_argument('--dropoutg', type=float, default=0.1, help='dropout for GNN layers')
    parser.add_argument('--dropoutf', type=float, default=0.2, help='dropout for fully-connected layers')

    # graph optimization
    parser.add_argument('-dlr', '--decoder_lr', default=DECODER_DEFAULT_LR[args.dataset], type=float, help='learning rate')
    parser.add_argument('-mbs', '--mini_batch_size', default=1, type=int)
    parser.add_argument('-ebs', '--eval_batch_size', default=4, type=int)
    parser.add_argument('--unfreeze_epoch', default=3, type=int)
    parser.add_argument('--refreeze_epoch', default=10000, type=int)

    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='show this help message and exit')
    args = parser.parse_args()
    if args.simple:
        parser.set_defaults(diag_decompose=True, gnn_layer_num=1, k=1)   

    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see
        # https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will
        # download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool(
        '.ckpt' in args.model_name_or_path), config=config)

    if args.use_consistency is True:
        model.set_lambda(args.lambda_a, args.lambda_b)

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will
        # download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    ###################################################################################################
    #   Load graph data                                                                               #
    ###################################################################################################
    if 'lm' in args.ent_emb:
        print('Using contextualized embeddings for concepts')
        use_contextualized = True
    else:
        use_contextualized = False
    print('use_contextualized: ', use_contextualized)
    cp_emb = [np.load(path) for path in args.ent_emb_paths]
    cp_emb = torch.tensor(np.concatenate(cp_emb, 1), dtype=torch.float)

    concept_num, concept_dim = cp_emb.size(0), cp_emb.size(1)
    print('| num_concepts: {} |'.format(concept_num))

    # Training
    if args.do_train:
        # if args.use_consistency is True:
        #     train_dataset = load_and_cache_examples_triplet(
        #         args, args.task_name, tokenizer, cp_emb, evaluate=False, random_sample=args.random_sample, use_contextualized=use_contextualized)
        # else:
        #     train_dataset = load_and_cache_examples(
        #         args, args.task_name, tokenizer, cp_emb,  evaluate=False, use_contextualized=use_contextualized)
        
        train_dataset = load_and_cache_examples(
                args, args.task_name, tokenizer, cp_emb,  evaluate=False, use_contextualized=use_contextualized)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s",
                    global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can
    # reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained
        # model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)

    # Evaluation
    results = {}
    if (args.do_eval or args.do_test) and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(
                glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(
                logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split(
                '-')[-1] if len(checkpoints) > 1 else ""
            # model = RobertaForSequenceClassification.from_pretrained(
            #     checkpoint)
            model = RobertaForSequenceClassificationConsistency.from_pretrained(
                checkpoint)
            model.to(args.device)

            result = evaluate(args, model, tokenizer, cp_emb, prefix=global_step)
            result = dict((k + '_{}'.format(global_step), v)
                          for k, v in result.items())
            results.update(result)
    print(results)

    return results


if __name__ == "__main__":
    main()
