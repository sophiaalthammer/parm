import os
import jsonlines
import json
import math
import ast
import pickle
import argparse
import pytrec_eval
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
sns.set(color_codes=True, font_scale=1.2)
from collections import Counter
from eval.eval_bm25 import read_run_whole_doc, analyze_correlations_bet_para, ranking_eval
from preprocessing.caselaw_stat_corpus import preprocess_label_file


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


def read_run_separate(bm25_folder: str, scores='ranks'):
    run = {}
    for root, dirs, files in os.walk(bm25_folder):
        for file in files:
            with open(os.path.join(bm25_folder, file), 'r') as f:
                lines = f.readlines()
                lines_dict = {}
                for i in range(len(lines)):
                    if scores == 'scores':
                        lines_dict.update({lines[i].split(' ')[0]: float(lines[i].split(' ')[-1].strip())})
                    else:
                        lines_dict.update({lines[i].split(' ')[0]: len(lines) - i})
                if run.get(file.split('_')[2]):
                    run.get(file.split('_')[2]).update({file.split('_')[3]: lines_dict})
                else:
                    run.update({file.split('_')[2]: {}})
                    run.get(file.split('_')[2]).update({file.split('_')[3]: lines_dict})
    return run



def read_run_separate_aggregate(bm25_folder: str, aggregation='interleave', scores='ranks'):
    # geh in den bm25 folder, lies in dokument und query: dann dict {query: {top 1000}}
    if aggregation == 'overlap_scores' or aggregation == 'mean_scores':
        scores = 'scores'

    run = read_run_separate(bm25_folder, scores)

    # now i need an aggregation function here, different choices
    if aggregation == 'overlap_docs':
        # now aggregate according to the overlap of the docs in the paragraphs!
        run_aggregated = aggregate_run_overlap(run)
    elif aggregation == 'interleave':
        run_aggregated = aggregate_run_interleave(run)
    elif aggregation == 'overlap_ranks':
        # now aggregate according to the overlap of the docs in the paragraphs!
        run_aggregated = aggregate_run_ranks_overlap(run)
    elif aggregation == 'rrf':
        # now aggregate according to the overlap of the docs in the paragraphs!
        run_aggregated = aggregate_run_rrf(run)
    elif aggregation == 'overlap_scores':
        run_aggregated = aggregate_run_ranks_overlap(run)
    elif aggregation == 'mean_scores':
        run_aggregated = aggregate_run_mean_score(run)
    if run_aggregated:
        return run_aggregated
    else:
        return run


def read_run_whole_doc(bm25_folder: str, scores='ranks'):
    # geh in den bm25 folder, lies in dokument und query: dann dict {query: {top 1000}}
    run = {}
    for root, dirs, files in os.walk(bm25_folder):
        for file in files:
            with open(os.path.join(bm25_folder, file), 'r') as f:
                lines = f.readlines()
                lines_dict = {}
                for i in range(len(lines)):
                    if scores == 'scores':
                        lines_dict.update({lines[i].split(' ')[0].split('_')[0]: float(lines[i].split(' ')[-1].strip())})
                    else:
                        lines_dict.update({lines[i].split(' ')[0].split('_')[0]: len(lines) - i})
                run.update({file.split('_')[2]: lines_dict})
    return run


def aggregate_run_overlap(run):
    for doc in run.keys():
        for para in run.get(doc).keys():
            for para_rel in run.get(doc).get(para).keys():
                run.get(doc).get(para)[para_rel] = 1
    run_aggregated = {}
    for doc in run.keys():
        for para in run.get(doc).keys():
            for para_rel, value in run.get(doc).get(para).items():
                if run_aggregated.get(doc):
                    if run_aggregated.get(doc).get('_'.join(para_rel.split('_')[:1])):
                        run_aggregated.get(doc).update({'_'.join(para_rel.split('_')[:1]): run_aggregated.get(
                            doc).get('_'.join(para_rel.split('_')[:1])) + 1})
                    else:
                        run_aggregated.get(doc).update({'_'.join(para_rel.split('_')[:1]): 1})
                else:
                    run_aggregated.update({doc: {}})
                    if run_aggregated.get(doc).get('_'.join(para_rel.split('_')[:1])):
                        run_aggregated.get(doc).update({'_'.join(para_rel.split('_')[:1]): run_aggregated.get(
                            doc).get('_'.join(para_rel.split('_')[:1])) + 1})
                    else:
                        run_aggregated.get(doc).update({'_'.join(para_rel.split('_')[:1]): 1})
    return run_aggregated


def aggregate_run_ranks_overlap(run):
    print(run)
    run_aggregated = {}
    for doc in run.keys():
        for para in run.get(doc).keys():
            for para_rel, value in run.get(doc).get(para).items():
                if run_aggregated.get(doc):
                    if run_aggregated.get(doc).get('_'.join(para_rel.split('_')[:1])):
                        run_aggregated.get(doc).update({'_'.join(para_rel.split('_')[:1]): run_aggregated.get(
                            doc).get('_'.join(para_rel.split('_')[:1])) + value})
                    else:
                        run_aggregated.get(doc).update({'_'.join(para_rel.split('_')[:1]): value})
                else:
                    run_aggregated.update({doc: {}})
                    if run_aggregated.get(doc).get('_'.join(para_rel.split('_')[:1])):
                        run_aggregated.get(doc).update({'_'.join(para_rel.split('_')[:1]): run_aggregated.get(
                            doc).get('_'.join(para_rel.split('_')[:1])) + value})
                    else:
                        run_aggregated.get(doc).update({'_'.join(para_rel.split('_')[:1]): value})
    return run_aggregated


def aggregate_run_rrf(run):
    run_aggregated = {}
    for doc in run.keys():
        for para in run.get(doc).keys():
            for para_rel, value in run.get(doc).get(para).items():
                if run_aggregated.get(doc):
                    if run_aggregated.get(doc).get('_'.join(para_rel.split('_')[:1])):
                        run_aggregated.get(doc).update({'_'.join(para_rel.split('_')[:1]): run_aggregated.get(
                            doc).get('_'.join(para_rel.split('_')[:1])) + (1/(60+(1001-value)))})
                    else:
                        run_aggregated.get(doc).update({'_'.join(para_rel.split('_')[:1]): (1/(60+(1001-value)))})
                else:
                    run_aggregated.update({doc: {}})
                    if run_aggregated.get(doc).get('_'.join(para_rel.split('_')[:1])):
                        run_aggregated.get(doc).update({'_'.join(para_rel.split('_')[:1]): run_aggregated.get(
                            doc).get('_'.join(para_rel.split('_')[:1])) + (1/(60+(1001-value)))})
                    else:
                        run_aggregated.get(doc).update({'_'.join(para_rel.split('_')[:1]): (1/(60+(1001-value)))})
    return run_aggregated


def aggregate_run_interleave(run):
    run_aggregated = {}
    for doc in run.keys():
        for para in run.get(doc).keys():
            for para_rel, value in run.get(doc).get(para).items():
                if run_aggregated.get(doc):
                    run_aggregated.get(doc).update({'_'.join(para_rel.split('_')[:1]): value})
                else:
                    run_aggregated.update({doc: {}})
                    run_aggregated.get(doc).update({'_'.join(para_rel.split('_')[:1]): value})
    return run_aggregated


def aggregate_run_mean_score(run):
    run_aggregated = {}
    for doc in run.keys():
        for para in run.get(doc).keys():
            for para_rel, value in run.get(doc).get(para).items():
                if run_aggregated.get(doc):
                    if run_aggregated.get(doc).get('_'.join(para_rel.split('_')[:1])):
                        run_aggregated.get(doc).get('_'.join(para_rel.split('_')[:1])).append(value)
                        #run_aggregated.get(doc).update({'_'.join(para_rel.split('_')[:2]): })
                    else:
                        run_aggregated.get(doc).update({'_'.join(para_rel.split('_')[:1]): [value]})
                else:
                    run_aggregated.update({doc: {}})
                    if run_aggregated.get(doc).get('_'.join(para_rel.split('_')[:1])):
                        run_aggregated.get(doc).get('_'.join(para_rel.split('_')[:1])).append(value)
                        #print(list)
                        #run_aggregated.get(doc).update({'_'.join(para_rel.split('_')[:2]): list})
                    else:
                        run_aggregated.get(doc).update({'_'.join(para_rel.split('_')[:1]): [value]})
    for doc in run_aggregated.keys():
        for key, item in run_aggregated.get(doc).items():
            run_aggregated.get(doc).update({key: np.mean(item)})
    return run_aggregated


def eval_ranking_bm25(label_file, bm25_folder, output_dir, output_file: str, aggregation='interleave', scores='ranks'):

    if aggregation == 'overlap_scores':
        scores = 'scores'

    qrels = read_label_file(label_file)
    if 'separate' in bm25_folder:
        run = read_run_separate_aggregate(bm25_folder, aggregation, scores)
    else:
        run = read_run_whole_doc(bm25_folder, scores)

    ranking_eval(qrels, run, output_dir, output_file)
    return run, qrels


def ranking_eval(qrels, run, output_dir, output_file= 'eval_bm25_aggregate_overlap.txt'):
    # trec eval
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'recall_100', 'recall_200', 'recall_300', 'recall_500', 'recall_1000', 'ndcg_cut_10', 'recip_rank'})

                                               #{'recall_1', 'recall_2', 'recall_3', 'recall_4', 'recall_5', 'recall_6', 'recall_7', 'recall_8',
                    #'recall_9', 'recall_10','recall_11', 'recall_12', 'recall_13', 'recall_14', 'recall_15', 'recall_16', 'recall_17', 'recall_18',
                    #'recall_19', 'recall_20','P_1', 'P_2', 'P_3', 'P_4', 'P_5', 'P_6', 'P_7', 'P_8', 'P_9', 'P_10',
                    #'P_11', 'P_12', 'P_13', 'P_14', 'P_15', 'P_16', 'P_17', 'P_18', 'P_19', 'P_20'}) # {'recall_100', 'recall_200', 'recall_300', 'recall_500', 'recall_1000'})

    results = evaluator.evaluate(run)

    def print_line(measure, scope, value):
        print('{:25s}{:8s}{:.4f}'.format(measure, scope, value))

    def write_line(measure, scope, value):
        return '{:25s}{:8s}{:.4f}'.format(measure, scope, value)

    for query_id, query_measures in sorted(results.items()):
        for measure, value in sorted(query_measures.items()):
            print_line(measure, query_id, value)

    #for measure in sorted(query_measures.keys()):
    #    print_line(
    #        measure,
    #        'all',
    #        pytrec_eval.compute_aggregated_measure(
    #            measure,
    #            [query_measures[measure]
    #             for query_measures in results.values()]))

        with open(os.path.join(output_dir, output_file), 'w') as output:
            for measure in sorted(query_measures.keys()):
                output.write(write_line(
                    measure,
                    'all',
                    pytrec_eval.compute_aggregated_measure(
                        measure,
                        [query_measures[measure]
                         for query_measures in results.values()])) + '\n')


def eval_ranking_overall_recall(label_file, bm25_folder, output_dir, mode, aggregation='interleave'):
    qrels = read_label_file(label_file)
    if 'separate' in bm25_folder:
        run = read_run_separate_aggregate(bm25_folder, aggregation)
    else:
        run = read_run_whole_doc(bm25_folder)

    # overall recall
    rec_per_topic = []
    for key, value in qrels.items():
        rel_run = run.get(key)
        value_list = [val for val in value.keys()]
        rel_run_list = [rel for rel in rel_run.keys()]
        rec_per_topic.append(len(list(set(value_list) & set(rel_run_list))) / len(value_list))

    print('Overall recall of {} is {}'.format(''.join([mode[0], mode[1]]), np.mean(rec_per_topic)))

    with open(os.path.join(output_dir, 'eval_recall_overall.txt'), 'w') as output:
        output.write('Overall recall of {} is {}'.format(''.join([mode[0], mode[1]]), np.mean(rec_per_topic)))


if __name__ == "__main__":
    #
    # config
    #

    #parser = argparse.ArgumentParser()

    #parser.add_argument('--label-file', action='store', dest='label_file',
    #                    help='org file with the guid and the labels', required=True)
    #parser.add_argument('--pred-file', action='store', dest='pred_file',
    #                    help='file with the binary prediction per guid', required=False)
    #parser.add_argument('--bm25-folder', action='store', dest='bm25_folder',
    #                    help='folder with the BM25 retrieval per guid which the result is compared to', required=False)
    #args = parser.parse_args()

    label_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/task1_train_2020_labels2.json'
    label_file_val = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/val/val_labels.json'
    label_file_test ='/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/test/task1_test_labels_2021.json'


    def eval_mode(mode):
        bm25_folder = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bm25/search/{}/{}'.format(mode[0], mode[1])
        output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bm25/eval/{}/{}'.format(mode[0], mode[1])

        if bm25_folder:
            if mode[0] == 'val':
                run, qrels = eval_ranking_bm25(label_file_val, bm25_folder, output_dir, 'eval_rrf2_bm25_aggregate_{}.txt'.format(mode[2]),aggregation=mode[2])
            elif mode[0] == 'train':
               run, qrels = eval_ranking_bm25(label_file, bm25_folder, output_dir, 'eval_rrf2_bm25_aggregate_{}.txt'.format(mode[2]), aggregation=mode[2])
            elif mode[0] == 'test':
               run, qrels = eval_ranking_bm25(label_file_test, bm25_folder, output_dir, 'eval_rrf2_bm25_aggregate_{}.txt'.format(mode[2]), aggregation=mode[2])
        return run, qrels


    def eval_all_bm25():
        ## train ##
        # whole doc evaluation
        #eval_mode(['train', 'whole_doc_para_only', 'overlap_docs'])
        #eval_mode(['train', 'whole_doc_w_summ_intro', 'overlap_docs'])

        # sep para: interleave
        #eval_mode(['train', 'separately_para_only', 'interleave'])
        #eval_mode(['train', 'separately_para_w_summ_intro', 'interleave'])

        # sep para: overlap docs
        #eval_mode(['train', 'separately_para_only', 'overlap_docs'])
        #eval_mode(['train', 'separately_para_w_summ_intro', 'overlap_docs'])

        # sep para: overlap ranks
        #eval_mode(['train', 'separately_para_only', 'overlap_ranks'])
        #eval_mode(['train', 'separately_para_w_summ_intro', 'overlap_ranks'])

        # sep para: overlap scores
        #eval_mode(['train', 'separately_para_only', 'overlap_scores'])
        #eval_mode(['train', 'separately_para_w_summ_intro', 'overlap_scores'])

        # sep para: mean scores
        #eval_mode(['train', 'separately_para_only', 'mean_scores'])
        #eval_mode(['train', 'separately_para_w_summ_intro', 'mean_scores'])


        ## val ##
        # whole doc evaluation
        #eval_mode(['val', 'whole_doc_para_only', 'overlap_docs'])
        #eval_mode(['val', 'whole_doc_w_summ_intro', 'overlap_docs'])

        # sep para: interleave
        #eval_mode(['val', 'separately_para_only', 'interleave'])
        #eval_mode(['val', 'separately_para_w_summ_intro', 'interleave'])

        # sep para: overlap docs
        #eval_mode(['val', 'separately_para_only', 'overlap_docs'])
        #eval_mode(['val', 'separately_para_w_summ_intro', 'overlap_docs'])

        # sep para: overlap ranks
        #eval_mode(['val', 'separately_para_only', 'overlap_ranks'])
        #eval_mode(['val', 'separately_para_w_summ_intro', 'overlap_ranks'])

        # sep para: overlap scores
        #eval_mode(['val', 'separately_para_only', 'overlap_scores'])
        #eval_mode(['val', 'separately_para_w_summ_intro', 'overlap_scores'])

        # sep para: mean scores
        #eval_mode(['val', 'separately_para_only', 'mean_scores'])
        #eval_mode(['val', 'separately_para_w_summ_intro', 'mean_scores'])

        ## test ##
        # whole doc evaluation
        #eval_mode(['test', 'whole_doc_para_only', 'overlap_docs'])
        #eval_mode(['test', 'whole_doc_w_summ_intro', 'overlap_docs'])

        # sep para: interleave
        #eval_mode(['test', 'separately_para_only', 'interleave'])
        #eval_mode(['test', 'separately_para_w_summ_intro', 'interleave'])

        # sep para: overlap docs
        #eval_mode(['test', 'separately_para_only', 'overlap_docs'])
        #eval_mode(['test', 'separately_para_w_summ_intro', 'overlap_docs'])

        # sep para: overlap ranks
        eval_mode(['test', 'separately_para_only', 'overlap_ranks'])
        eval_mode(['test', 'separately_para_w_summ_intro', 'overlap_ranks'])

        # sep para: overlap scores
        #eval_mode(['test', 'separately_para_only', 'overlap_scores'])
        #eval_mode(['test', 'separately_para_w_summ_intro', 'overlap_scores'])

        # sep para: mean scores
        #eval_mode(['test', 'separately_para_only', 'mean_scores'])
        #eval_mode(['test', 'separately_para_w_summ_intro', 'mean_scores'])


    #run, qrels = eval_mode(['test', 'whole_doc_w_summ_intro', 'rrf'])

    #output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bm25/aggregate/test/separately_para_w_summ_intro/'
    #with open(os.path.join(output_dir, 'run_aggregated_test_whole_doc_overlap_docs.pickle'), 'wb') as f:
    #    pickle.dump(run, f)

    #run2, qrels2 = eval_mode(['test', 'separately_para_w_summ_intro', 'rrf'])
    #with open(os.path.join(output_dir, 'run_aggregated_test_rrf_overlap_ranks.pickle'), 'wb') as f:
    #    pickle.dump(run2, f)

    #eval_all_bm25()

    ## evaluate overall recall
    #mode = ['train', 'whole_doc_w_summ_intro', 'overlap_docs']

    #bm25_folder = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bm25/search/{}/{}'.format(mode[0], mode[1])
    #output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bm25/eval/{}/{}'.format(mode[0], mode[1])

    # evaluate overall recall: better for separate retrieval -> then aggregation function needs to be changed!
    #if bm25_folder:
    #    if mode[0] == 'val':
    #        eval_ranking_overall_recall(label_file_val, bm25_folder, output_dir, mode)
    #    elif mode[0] == 'train':
    #        eval_ranking_overall_recall(label_file, bm25_folder, output_dir, mode)


    ## analyze correlation between query and document paragraphs
    mode = ['test', 'separately_para_w_summ_intro', 'overlap_ranks']
    bm25_folder = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bm25/search/{}/{}'.format(mode[0], mode[1])
    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bm25/eval/{}/{}'.format(mode[0], mode[1])

    # plot heatmap bm25, works!
    run = read_run_separate(bm25_folder)
    qrels = read_label_file('/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/test/task1_test_labels_2021.json')
    #analyze_correlations_bet_para(run, output_dir)

    # run leave only relevant docs inside, and then let it run through analysis
    run_rel = {}
    for key, value in run.items():
        run_rel.update({key:{}})
        rel_id = list(qrels.get(key).keys())
        for key2, value2 in value.items():
            run_rel.get(key).update({key2:{}})
            for key3, value3 in value2.items():
                if key3.split('_')[0] in rel_id:
                    run_rel.get(key).get(key2).update({key3:value3})

    pd_interactions = analyze_correlations_bet_para(run_rel, output_dir)

    print(pd_interactions)

    # column wise sum: so how many candidate documents got found by this paragraphs
    print(pd_interactions.sum(axis=0))

    # row wise sum
    print(pd_interactions.sum(axis=1))















