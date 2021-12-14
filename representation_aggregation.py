import os
import pathlib
import json
import argparse
import csv
import sys
csv.field_size_limit(sys.maxsize)
import pickle

import numpy as np

from eval.eval_bm25_coliee2021 import read_label_file, ranking_eval
from eval.eval_dpr_coliee2021 import read_run_separate


def read_encoded_ctx_file(encoded_ctx_file: str):
    """
    Returns dictionary containing the encoded passages and their vector embeddings
    :param encoded_ctx_file:
    :return:
    """
    with open(encoded_ctx_file, mode="rb") as f:
        p_emb = pickle.load(f)

    # create dictionary, so that i can lookup embeddings
    p_emb_dict = {}
    for passage in p_emb:
        p_emb_dict.update({passage[0]: passage[1]})
    return p_emb_dict


def read_encoded_qa_file(encoded_qa_file: str):
    """
    Returns dictionary containing the encoded queries and their vector embeddings
    :param encoded_qa_file:
    :return:
    """
    # now i need the query encoder
    with open(encoded_qa_file, mode="rb") as f:
        q_emb = pickle.load(f)

    # create dictionary, so that i can lookup embeddings
    q_emb_dict = {}
    for passage in q_emb:
        q_emb_dict.update({passage[0][0]: passage[1]})
    return q_emb_dict


def dict_ids_with_embeddings(q_emb_dict):
    # dictionary with query_id and all the embeddings to it
    q_ids_w_emb = {}
    for key, value in q_emb_dict.items():
        query_id = key.split('_')[0]
        if q_ids_w_emb.get(query_id):
            embeddings = q_ids_w_emb.get(query_id)
            embeddings.append(value)
            q_ids_w_emb.update({query_id: embeddings})
        else:
            q_ids_w_emb.update({query_id: [value]})
    return q_ids_w_emb


def aggregate_emb_avg(q_ids_w_emb: dict):
    # avg
    q_ids_avg_emb = {}
    for key, value in q_ids_w_emb.items():
        q_ids_avg_emb.update({key: np.mean(value, axis=0)})
    return q_ids_avg_emb


def aggregate_emb_sum(q_ids_w_emb: dict):
    # avg
    q_ids_avg_emb = {}
    for key, value in q_ids_w_emb.items():
        q_ids_avg_emb.update({key: np.sum(value, axis=0)})
    return q_ids_avg_emb


def aggregate_emb_max(q_ids_w_emb: dict):
    # avg
    q_ids_avg_emb = {}
    for key, value in q_ids_w_emb.items():
        q_ids_avg_emb.update({key: np.max(value, axis=0)})
    return q_ids_avg_emb


def aggregate_emb_min(q_ids_w_emb: dict):
    # avg
    q_ids_avg_emb = {}
    for key, value in q_ids_w_emb.items():
        q_ids_avg_emb.update({key: np.min(value, axis=0)})
    return q_ids_avg_emb


def aggregate_emb_scores(q_ids_w_emb: dict):
    p_ids_avg_emb = {}
    for key, value in q_ids_w_emb.items():
        list_emb = [emb[0] for emb in value]
        list_weights = [emb[1] for emb in value]
        p_ids_avg_emb.update({key: np.dot(list_weights, list_emb)})
    return p_ids_avg_emb


def aggregate_emb_vrrf(q_ids_w_emb: dict):
    p_ids_avg_emb = {}
    for key, value in q_ids_w_emb.items():
        list_emb = [emb[0] for emb in value]
        list_weights = [(1/(60+(1001- emb[1]))) for emb in value]
        p_ids_avg_emb.update({key: np.dot(list_weights, list_emb)})
    return p_ids_avg_emb


def aggregate_ids_with_embeddings(q_ids_w_emb: dict, aggregation_mode: str):
    """
    Aggregates the embeddings according to the aggregation mode
    :param q_ids_w_emb:
    :return:
    """
    if aggregation_mode == 'avg':
        q_ids_agg_emb = aggregate_emb_avg(q_ids_w_emb)
    elif aggregation_mode == 'sum':
        q_ids_agg_emb = aggregate_emb_sum(q_ids_w_emb)
    elif aggregation_mode == 'max':
        q_ids_agg_emb = aggregate_emb_max(q_ids_w_emb)
    elif aggregation_mode == 'min':
        q_ids_agg_emb = aggregate_emb_min(q_ids_w_emb)
    elif aggregation_mode == 'vrrf':
        q_ids_agg_emb = aggregate_emb_vrrf(q_ids_w_emb)
    elif aggregation_mode == 'vscores':
        q_ids_agg_emb = aggregate_emb_scores(q_ids_w_emb)
    elif aggregation_mode == 'vranks':
        q_ids_agg_emb = aggregate_emb_scores(q_ids_w_emb)
    else:
        print('No valid aggregation mode entered')
        return None
    return q_ids_agg_emb


def aggregate_p_in_run(run: dict, p_emb_dict: dict):
    """
    aggregates passages from run
    :param run:
    :param p_emb_dict:
    :return:
    """
    run_p_embs = {}
    for q_id, retrieved_lists in run.items():
        run_p_embs.update({q_id: {}})
        for q_p_number, ranked_list in retrieved_lists.items():
            for p_id, score in ranked_list.items():
                p_emb = p_emb_dict.get(p_id)
                p_id_short = p_id.split('_')[0]
                if run_p_embs.get(q_id).get(p_id_short):
                    list_emb = run_p_embs.get(q_id).get(p_id_short)
                    list_emb.append(p_emb)
                    run_p_embs.get(q_id).update({p_id_short: list_emb})
                else:
                    run_p_embs.get(q_id).update({p_id_short: [p_emb]})
    return run_p_embs


def aggregate_run_in_p_with_scores(run: dict, p_emb_dict: dict):
    run_p_embs = {}
    for q_id, retrieved_lists in run.items():
        run_p_embs.update({q_id: {}})
        for q_p_number, ranked_list in retrieved_lists.items():
            for p_id, score in ranked_list.items():
                p_emb = p_emb_dict.get(p_id)
                p_id_short = p_id.split('_')[0]
                if run_p_embs.get(q_id).get(p_id_short):
                    list_emb = run_p_embs.get(q_id).get(p_id_short)
                    list_emb.append((p_emb, score))
                    run_p_embs.get(q_id).update({p_id_short: list_emb})
                else:
                    run_p_embs.get(q_id).update({p_id_short: [(p_emb, score)]})
    return run_p_embs


def aggregate_passage_embeddings_in_run(run: dict, p_emb_dict: dict, aggregation_mode: str):
    # first i do only from the ranked lists
    if aggregation_mode=='vrrf' or aggregation_mode=='vranks' or aggregation_mode=='vscores':
        run_p_embs = aggregate_run_in_p_with_scores(run, p_emb_dict)
    else:
        run_p_embs = aggregate_p_in_run(run, p_emb_dict)

    # now for each document, merge the same documents which overlap and then average/sum/min/max
    run_q_id_p_id_aggregated = {}
    for q_id, retrieved_lists in run_p_embs.items():
        p_ids_agg_emb = aggregate_ids_with_embeddings(retrieved_lists, aggregation_mode)
        run_q_id_p_id_aggregated.update({q_id: p_ids_agg_emb})
    return run_q_id_p_id_aggregated


def aggregate_passage_embeddings_whole_doc(run: dict, p_emb_dict: dict, aggregation_mode: str):
    p_ids_w_emb = dict_ids_with_embeddings(p_emb_dict)
    p_ids_agg_emb = aggregate_ids_with_embeddings(p_ids_w_emb, aggregation_mode)

    # now add this to the run, so that i have the same representation as for the first option
    run_pd_id_emb_agg = {}
    for q_id, retrieved_lists in run.items():
        run_pd_id_emb_agg.update({q_id: {}})
        for q_p_number, ranked_list in retrieved_lists.items():
            for p_id, score in ranked_list.items():
                p_id_short = p_id.split('_')[0]
                run_pd_id_emb_agg.get(q_id).update({p_id_short: p_ids_agg_emb.get(p_id_short)})
    return run_pd_id_emb_agg


def score_run_dot_product(run_pd_id_emb_agg: dict, q_ids_agg_emb: dict):
    """
    Score the runs with the dot product between the query and candidate embedding
    :param run_pd_id_emb_agg:
    :param q_ids_agg_emb:
    :return:
    """
    run_scores_aggregated_emb = {}
    for q_id, retrieved_list in run_pd_id_emb_agg.items():
        run_scores_aggregated_emb.update({q_id: {}})
        q_emb = q_ids_agg_emb.get(q_id)
        for candidate_id, candidate_emb in retrieved_list.items():
            # normally i dont want to have int here... but maybe then trec eval works...
            run_scores_aggregated_emb.get(q_id).update({candidate_id: int(np.vdot(q_emb, candidate_emb))})

    run_scores_aggregated_emb_sorted = {}
    for q_id, run in run_scores_aggregated_emb.items():
        run_scores_aggregated_emb_sorted.update(
            {q_id: {k: v for k, v in sorted(run.items(), key=lambda item: item[1], reverse=True)}})
    return run_scores_aggregated_emb_sorted


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
    # now open the output to create the matchings
    if aggregation_mode == 'vrrf' or aggregation_mode == 'vranks':
        run = read_run_separate(output_file, scores="ranks")
    else:
        run = read_run_separate(output_file, scores="scores")
    qrels = read_label_file(label_file)

    # different pooling strategies: pool the query and passage document representation
    # passage aggregation: first pool independently, then maybe pool with interaction? lets see...
    # first for the query
    q_ids_w_emb = dict_ids_with_embeddings(q_emb_dict)
    if aggregation_mode=='vscores' or aggregation_mode=='vrrf' or aggregation_mode=='vranks':
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
    run_scores_emb_agg = score_run_dot_product(run_pd_id_emb_agg, q_ids_agg_emb)

    with open(os.path.join(output_dir, 'run_aggregated_test_{}.pickle'.format(aggregation_mode)), 'wb') as f:
        pickle.dump(run_scores_emb_agg, f)

    # then evaluate run with evaluation of whole document runs right?
    #ranking_eval(qrels, run_scores_emb_agg, output_dir, output_file_name)


def main(args):
    aggregate_eval(args.encoded_ctx_file, args.encoded_qa_file, args.output_top1000s, args.label_file,
                   args.aggregation_mode, args.candidate_mode, args.output_dir, args.output_file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--encoded_ctx_file",
        required=True,
        type=str,
        default=None,
        help="Path to the encoded ctx file, with the encodings of the passages in the corpus"
    )
    parser.add_argument(
        "--encoded_qa_file",
        required=True,
        type=str,
        default=None,
        help="Path to the encoded qa file, with the encodings of the query passages",
    )
    parser.add_argument(
        "--output_top1000s",
        required=True,
        type=str,
        default=None,
        help="Path to the output file from the dense_retriever.py search with the topNs result",
    )
    parser.add_argument(
        "--label_file",
        required=True,
        type=str,
        default=None,
        help="Path to the label file",
    )
    parser.add_argument(
        "--aggregation_mode",
        required=True,
        type=str,
        default='sum',
        choices = ['sum', 'avg', 'max', 'min', 'cnn', 'ffn', 'trans', 'vrrf', 'vscores', 'vranks'],
        help="Aggregation mode for aggregating the embedding representations of query and candidate documents:"
             "choose between sum/max/min/avg/cnn/ffn/trans",
    )
    parser.add_argument(
        "--candidate_mode",
        required=True,
        type=str,
        default='p_from_retrieved_list',
        choices=['p_from_retrieved_list', 'p_from_whole_doc'],
        help="Determines which paragraph embeddings to choose for aggregation of the candidate document embedding:"
             "either take only the embeddings from the passages from the retrieved list or take the embeddings from all passages in the candidate document",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        default=None,
        help="Path to the output directory where the evaluation will be stored",
    )
    parser.add_argument(
        "--output_file_name",
        required=True,
        type=str,
        default='eval.txt',
        help="Name of the file containing the evaluation measures",
    )

    args = parser.parse_args()

    main(args)

    # mode = ['val', 'vrrf', 'p_from_retrieved_list']
    # encoded_ctx_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/dpr/legal_task2/legalbert/encoded_ctx_file/ctx_separate_para_dense2_0.pkl'
    # encoded_qa_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/dpr/legal_task2/legalbert/encoded_qa_file/{}_separate_para_questions_tensors.pkl'.format(
    #     mode[0])
    # output_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/dpr/legal_task2/legalbert/output/{}/{}_separate_para_top1000.json'.format(
    #     mode[0], mode[0])
    #
    # if mode[0] == 'train':
    #     label_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/train/train_wo_val_labels.json'
    # elif mode[0] == 'val':
    #     label_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/val/val_labels.json'
    # elif mode[0] == 'test':
    #     label_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/test/task1_test_labels_2021.json'
    # else:
    #     raise ValueError('No valid mode chosen, choose between train, val and test')
    #
    # aggregation_mode = mode[1]
    # candidate_mode = mode[2]
    #
    # output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/dpr/legal_task2/legalbert/eval/{}'.format(
    #     mode[0])
    # output_file_name = 'eval_dpr_aggregate_embeddings_{}_aggregation_{}.txt'.format(mode[0], aggregation_mode)
    #
    # p_emb_dict = read_encoded_ctx_file(encoded_ctx_file)
    # q_emb_dict = read_encoded_qa_file(encoded_qa_file)
    # # now open the output to create the matchings
    # if aggregation_mode=='vrrf' or aggregation_mode=='vranks':
    #     run = read_run_separate(output_file, scores="ranks")
    # else:
    #     run = read_run_separate(output_file, scores="scores")
    #
    # qrels = read_label_file(label_file)
    #
    # # different pooling strategies: pool the query and passage document representation
    # # passage aggregation: first pool independently, then maybe pool with interaction? lets see...
    # # first for the query
    # q_ids_w_emb = dict_ids_with_embeddings(q_emb_dict)
    # q_ids_agg_emb = aggregate_ids_with_embeddings(q_ids_w_emb, aggregation_mode)
    #
    # # then aggregate the candidate documents
    # # stop i can only aggregate the embeddings for the passages which got retrieved by one document!
    # # so i also need to take into account the run and the top1000s
    # # two possibilities: take the embeddings which are only in the retrieved list
    # # or take the document embedding from the whole corpus
    #
    # # first i do only from the ranked lists
    # if candidate_mode == 'p_from_retrieved_list':
    #     run_pd_id_emb_agg = aggregate_passage_embeddings_in_run(run, p_emb_dict, aggregation_mode)
    #
    #     if aggregation_mode == 'vrrf' or aggregation_mode == 'vranks' or aggregation_mode == 'vscores':
    #         run_p_embs = {}
    #         for q_id, retrieved_lists in run.items():
    #             run_p_embs.update({q_id: {}})
    #             for q_p_number, ranked_list in retrieved_lists.items():
    #                 for p_id, score in ranked_list.items():
    #                     p_emb = p_emb_dict.get(p_id)
    #                     p_id_short = p_id.split('_')[0]
    #                     if run_p_embs.get(q_id).get(p_id_short):
    #                         list_emb = run_p_embs.get(q_id).get(p_id_short)
    #                         list_emb.append((p_emb, score))
    #                         run_p_embs.get(q_id).update({p_id_short: list_emb})
    #                     else:
    #                         run_p_embs.get(q_id).update({p_id_short: [(p_emb, score)]})
    #         run_p_embs = aggregate_run_in_p_with_scores(run, p_emb_dict)
    #     else:
    #         run_p_embs = aggregate_p_in_run(run, p_emb_dict)
    #
    #     # now for each document, merge the same documents which overlap and then average/sum/min/max
    #     run_q_id_p_id_aggregated = {}
    #     for q_id, retrieved_lists in run_p_embs.items():
    #         #p_ids_agg_emb = aggregate_ids_with_embeddings(retrieved_lists, aggregation_mode)
    #         p_ids_avg_emb = {}
    #         for key, value in retrieved_lists.items():
    #             list_emb = [emb[0] for emb in value]
    #             list_weights = [(1 / (60 + (1001 - emb[1]))) for emb in value]
    #             print(np.dot(list_weights, list_emb).shape())
    #             p_ids_avg_emb.update({key: np.dot(list_weights, list_emb)})
    #
    #         run_q_id_p_id_aggregated.update({q_id: p_ids_avg_emb})
    #
    # # i could also make a weighting: how many passages of the document got retrieved, how many not
    # # (or influence on the representation of the retrieved passages on the overall representation)
    # # or include homogeneity (how many passages got retrieved)
    #
    # # or weighting of the ranks of the embeddings of the passages! then the more often!
    # # embedding multiply with the reciprical rank of the document! then the overlap and the embedding
    #
    # # this is the second option where the candidate document embedding consists of all passages
    # elif candidate_mode == 'p_from_whole_doc':
    #     run_pd_id_emd_agg = aggregate_passage_embeddings_whole_doc(run, p_emb_dict, aggregation_mode)
    # # will only work with a harder cutoff i think... maybe @200 or 500
    #
    # # now i have the aggregated embeddings of the query document and the candidate documents in the run form
    # # now score run with the dot product of the embeddings
    # run_scores_emb_agg = score_run_dot_product(run_pd_id_emb_agg, q_ids_agg_emb)
    #
    # # then evaluate run with evaluation of whole document runs right?
    # ranking_eval(qrels, run_scores_emb_agg, output_dir, output_file_name)





    # learn: ffn, cnn, transformer? learn on the retrieved embeddings, not on the whole doc


    # but i could take into account how many other passages the document has, which did not get retrieved (somehow like bm25 on the passages!)
    # which passages got retrieved, how many other passages got retrieved, something like that... to take into account the homogeneity of the doc
    # instead of RRF!
    # try it out!

    # der pool ist ja schon query dependent also kann dann auch die aggregierung query dependent sein
    # bei parade genau dasselbe


    # überlegungen etwas auf der dokumentenebene zu trainieren
    # learn cnn, ffn, transformer, svm on the representations also for document ranking? does that make sense?
    # begin with svm, labels from the qrels, and embeddings from the output files!

