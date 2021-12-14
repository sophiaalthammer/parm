import os
import pathlib
import json
import argparse
import csv
import sys
csv.field_size_limit(sys.maxsize)
import logging
import pickle
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from dpr.models import init_biencoder_components
from dpr.options import (
    add_encoder_params,
    setup_args_gpu,
    print_args,
    set_encoder_params_from_state,
    add_tokenizer_params,
    add_cuda_params,
)
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
    move_to_device,
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)


def read_run_whole_doc(pred_dir: str, scores='ranks'):
    with open(pred_dir, 'r') as json_file:
        pred = json.load(json_file)

    pred_dict = {}
    for question in pred:
        question_id = question.get('answers')[0].split('_')[0]
        pred_list = {}
        i = 0
        for predition in question.get('ctxs'):
            if scores == 'scores':
                pred_list.update({predition.get('id'): float(predition.get('score'))})
            else:
                pred_list.update({predition.get('id'): len(question.get('ctxs')) - i})
                i += 1
        pred_dict.update({question_id: pred_list})
    return pred_dict


def read_run_separate(pred_dir: str, scores='ranks'):
    with open(pred_dir, 'r') as json_file:
        pred = json.load(json_file)

    pred_dict = {}
    for question in pred:
        question_id = question.get('answers')[0].split('_')[0]
        question_para_id = question.get('answers')[0].split('_')[1]
        pred_list = {}
        i = 0
        for prediction in question.get('ctxs'):
            if scores == 'scores':
                pred_list.update({prediction.get('id'): float(prediction.get('score'))})
            else:
                pred_list.update({prediction.get('id'): len(question.get('ctxs')) - i})
                i += 1
        if pred_dict.get(question_id):
            pred_dict.get(question_id).update({question_para_id: pred_list})
        else:
            pred_dict.update({question_id: {}})
            pred_dict.get(question_id).update({question_para_id: pred_list})
    return pred_dict


def read_label_file(label_file: str):
    with open(label_file, 'rb') as f:
        labels = json.load(f)

    # other format of labels:
    qrels = {}
    for key, values in labels.items():
        val_format = {}
        for value in values:
            val_format.update({'{}'.format(value.split('.')[0]): 1})
        qrels.update({key.split('.')[0]: val_format})
    return qrels


encoded_ctx_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/dpr/legal_task2/legalbert/encoded_ctx_file/ctx_separate_para_dense2_0.pkl'
encoded_qa_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/dpr/legal_task2/legalbert/encoded_qa_file/val_separate_para_questions_tensors.pkl'
output_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/dpr/legal_task2/legalbert/output/val/val_separate_para_top1000_2.json'

label_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/train/train_wo_val_labels.json'
label_file_val = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/val/val_labels.json'
label_file_test = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/test/task1_test_labels_2021.json'


with open(encoded_ctx_file, mode="rb") as f:
    p_emb = pickle.load(f)
#print(p_emb[0][1])

# create dictionary, so that i can lookup embeddings
p_emb_dict = {}
for passage in p_emb:
    p_emb_dict.update({passage[0]: passage[1]})

# now i need the query encoder
with open(encoded_qa_file, mode="rb") as f:
    q_emb = pickle.load(f)

# create dictionary, so that i can lookup embeddings
q_emb_dict = {}
for passage in q_emb:
    q_emb_dict.update({passage[0][0]: passage[1]})


# now open the output to create the matchings
run = read_run_whole_doc(output_file, scores="scores")
qrels = read_label_file(label_file_val)

# different pooling strategies


# then similarity with dot product


# then evaluate


# pool the query and passage document representation

# passage aggregation!
# pool with: sum, avg, max



# learn: ffn, cnn, transformer?

# der pool ist ja schon query dependent also kann dann auch die aggregierung query dependent sein
# bei parade genau dasselbe


# überlegungen etwas auf der dokumentenebene zu trainieren
# learn cnn, ffn, transformer, svm on the representations also for document ranking? does that make sense?
# begin with svm, labels from the qrels, and embeddings from the output files!

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#
#     add_encoder_params(parser)
#     add_tokenizer_params(parser)
#     add_cuda_params(parser)
#
#     parser.add_argument(
#         "--ctx_file", type=str, default=None, help="Path to passages set .tsv file"
#     )
#     parser.add_argument(
#         "--out_file",
#         required=True,
#         type=str,
#         default=None,
#         help="output .tsv file path to write results to ",
#     )
#     parser.add_argument(
#         "--shard_id",
#         type=int,
#         default=0,
#         help="Number(0-based) of data shard to process",
#     )
#     parser.add_argument(
#         "--num_shards", type=int, default=1, help="Total amount of data shards"
#     )
#     parser.add_argument(
#         "--batch_size",
#         type=int,
#         default=32,
#         help="Batch size for the passage encoder forward pass",
#     )
#     args = parser.parse_args()
#
#     assert (
#         args.model_file
#     ), "Please specify --model_file checkpoint to init model weights"
# #
# #     setup_args_gpu(args)
#
#     main(args)