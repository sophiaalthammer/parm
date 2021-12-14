import os
import pathlib
import json
import argparse
import csv
import numpy as np
import sys
csv.field_size_limit(sys.maxsize)
import pickle

from eval.eval_bm25_coliee2021 import ranking_eval
from preprocessing.caselaw_stat_corpus import preprocess_label_file
from representation_aggregation import read_encoded_ctx_file, read_encoded_qa_file, dict_ids_with_embeddings, \
    aggregate_ids_with_embeddings, aggregate_passage_embeddings_in_run, aggregate_passage_embeddings_whole_doc,\
    score_run_dot_product


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
                pred_list.update({prediction.get('id'): float(len(question.get('ctxs')) - i)})
                i += 1
        if pred_dict.get(question_id):
            pred_dict.get(question_id).update({question_para_id: pred_list})
        else:
            pred_dict.update({question_id: {}})
            pred_dict.get(question_id).update({question_para_id: pred_list})
    return pred_dict


def aggregate_eval(encoded_ctx_file, encoded_qa_file, output_file, label_file, aggregation_mode, candidate_mode, output_dir, output_file_name):
    """
    reads in embeddings from query and candidate file, aggregates them according to the candidate and aggregation mode
    and evaluates the ranking
    :param encoded_ctx_file:
    :param encoded_qa_file:
    :param output_file:
    :param label_file:
    :param candidate_mode:
    :param aggregation_mode:
    :return:
    """
    p_emb_dict = read_encoded_ctx_file(encoded_ctx_file)
    q_emb_dict = read_encoded_qa_file(encoded_qa_file)
    q_emb_dict_updated = {}
    for key, value in q_emb_dict.items():
        q_emb_dict_updated.update({key.replace('_', '', 1): value})
    q_emb_dict = q_emb_dict_updated
    # now open the output to create the matchings
    if aggregation_mode == 'vrrf' or aggregation_mode == 'vranks':
        run = read_run_separate(output_file, scores="ranks")
    else:
        run = read_run_separate(output_file, scores="scores")

    qrels = preprocess_label_file(label_file)
    qrels_updated = {}
    for key, value in qrels.items():
        qrels_updated.update({key: {}})
        for val in value:
            qrels_updated.get(str(key)).update({str(val): 1})

    # different pooling strategies: pool the query and passage document representation
    # passage aggregation: first pool independently, then maybe pool with interaction? lets see...
    # first for the query
    q_ids_w_emb = dict_ids_with_embeddings(q_emb_dict)
    if aggregation_mode == 'vscores' or aggregation_mode == 'vrrf' or aggregation_mode == 'vranks':
        q_ids_agg_emb = aggregate_ids_with_embeddings(q_ids_w_emb, 'sum')
    else:
        q_ids_agg_emb = aggregate_ids_with_embeddings(q_ids_w_emb, aggregation_mode)

    # then aggregate the candidate documents
    # stop i can only aggregate the embeddings for the passages which got retrieved by one document!
    # so i also need to take into account the run and the top1000s
    # two possibilities: take the embeddings which are only in the retrieved list
    # or take the document embedding from the whole corpus

    # first i do only from the ranked lists
    if candidate_mode == 'p_from_retrieved_list':
        run_pd_id_emb_agg = aggregate_passage_embeddings_in_run(run, p_emb_dict, aggregation_mode)

    # i could also make a weighting: how many passages of the document got retrieved, how many not
    # (or influence on the representation of the retrieved passages on the overall representation)
    # or include homogeneity (how many passages got retrieved)

    # or weighting of the ranks of the embeddings of the passages! then the more often!
    # embedding multiply with the reciprical rank of the document! then the overlap and the embedding

    # this is the second option where the candidate document embedding consists of all passages
    elif candidate_mode == 'p_from_whole_doc':
        run_pd_id_emd_agg = aggregate_passage_embeddings_whole_doc(run, p_emb_dict, aggregation_mode)
    # will only work with a harder cutoff i think... maybe @200 or 500

    # now i have the aggregated embeddings of the query document and the candidate documents in the run form
    # now score run with the dot product of the embeddings
    run_scores_aggregated_emb = {}
    for q_id, retrieved_list in run_pd_id_emb_agg.items():
        q_emb = q_ids_agg_emb.get(str(q_id))
        if q_emb is not None:
            run_scores_aggregated_emb.update({q_id: {}})
            for candidate_id, candidate_emb in retrieved_list.items():
                # print(candidate_emb)
                # normally i dont want to have int here... but maybe then trec eval works...
                run_scores_aggregated_emb.get(q_id).update({candidate_id: int(np.vdot(q_emb, candidate_emb))})

    run_scores_aggregated_emb_sorted = {}
    for q_id, run in run_scores_aggregated_emb.items():
        run_scores_aggregated_emb_sorted.update(
            {q_id: {k: v for k, v in sorted(run.items(), key=lambda item: item[1], reverse=True)}})
    run_scores_emb_agg = run_scores_aggregated_emb_sorted

    with open(os.path.join(output_dir, 'run_aggregated_train_{}.pickle'.format(aggregation_mode)), 'wb') as f:
        pickle.dump(run_scores_emb_agg, f)
    with open(os.path.join(output_dir, 'qrels_train.pickle'), 'wb') as f:
        pickle.dump(qrels_updated, f)

    # then evaluate run with evaluation of whole document runs right?
    ranking_eval(qrels, run_scores_emb_agg, output_dir, output_file_name)


def main(args):
    aggregate_eval(args.encoded_ctx_file, args.encoded_qa_file, args.output_top1000s, args.label_file,
                   args.aggregation_mode, args.candidate_mode, args.output_dir, args.output_file_name)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument(
    #     "--encoded_ctx_file",
    #     required=True,
    #     type=str,
    #     default=None,
    #     help="Path to the encoded ctx file, with the encodings of the passages in the corpus"
    # )
    # parser.add_argument(
    #     "--encoded_qa_file",
    #     required=True,
    #     type=str,
    #     default=None,
    #     help="Path to the encoded qa file, with the encodings of the query passages",
    # )
    # parser.add_argument(
    #     "--output_top1000s",
    #     required=True,
    #     type=str,
    #     default=None,
    #     help="Path to the output file from the dense_retriever.py search with the topNs result",
    # )
    # parser.add_argument(
    #     "--label_file",
    #     required=True,
    #     type=str,
    #     default=None,
    #     help="Path to the label file",
    # )
    # parser.add_argument(
    #     "--aggregation_mode",
    #     required=True,
    #     type=str,
    #     default='sum',
    #     choices = ['sum', 'avg', 'max', 'min', 'cnn', 'ffn', 'trans', 'vrrf', 'vscores', 'vranks'],
    #     help="Aggregation mode for aggregating the embedding representations of query and candidate documents:"
    #          "choose between sum/max/min/avg/cnn/ffn/trans",
    # )
    # parser.add_argument(
    #     "--candidate_mode",
    #     required=True,
    #     type=str,
    #     default='p_from_retrieved_list',
    #     choices=['p_from_retrieved_list', 'p_from_whole_doc'],
    #     help="Determines which paragraph embeddings to choose for aggregation of the candidate document embedding:"
    #          "either take only the embeddings from the passages from the retrieved list or take the embeddings from all passages in the candidate document",
    # )
    # parser.add_argument(
    #     "--output_dir",
    #     required=True,
    #     type=str,
    #     default=None,
    #     help="Path to the output directory where the evaluation will be stored",
    # )
    # parser.add_argument(
    #     "--output_file_name",
    #     required=True,
    #     type=str,
    #     default='eval.txt',
    #     help="Name of the file containing the evaluation measures",
    # )
    #
    # args = parser.parse_args()
    #
    # main(args)

    mode = ['train', 'vrrf', 'p_from_retrieved_list', 'legalbert']
    encoded_ctx_file = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/dpr/{}/encoded_ctx_file/ctx_separate_para_dense_0.pkl'.format(mode[3])
    encoded_qa_file = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/dpr/{}/encoded_qa_file/train_separate_para_questions_tensors.pkl'.format(mode[3])
    output_file = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/dpr/{}/output/{}_separate_para_top1000.json'.format(mode[3], mode[0])

    if mode[0] == 'train':
        label_file = '/mnt/c/Users/salthamm/Documents/coding/ussc-caselaw-collection/airs2017-collection/qrel.txt'
    else:
        raise ValueError('No valid mode chosen, choose between train, val and test')

    aggregation_mode = mode[1]

    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/dpr/{}/eval/vector'.format(mode[3])
    output_file_name = 'eval_dpr_aggregate_embeddings_{}_aggregation_{}.txt'.format(mode[0], aggregation_mode)

    with open(os.path.join(output_dir, 'run_aggregated_train_{}.pickle'.format(aggregation_mode)), 'rb') as f:
        run = pickle.load(f)
    with open(os.path.join(output_dir, 'qrels_train.pickle'), 'rb') as f:
        qrels = pickle.load(f)

    # then evaluate run with evaluation of whole document runs right?
    ranking_eval(qrels2, run2, output_dir, output_file_name)

