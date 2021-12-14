import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from eval.eval_bm25_coliee2021 import read_label_file #ranking_eval
from retrieval.bm25_aggregate_paragraphs import sort_write_trec_output
from preprocessing.coliee21_task2_bm25 import ranking_eval
from analysis.compare_bm25_dpr import read_in_aggregated_scores, evaluate_weight, plot_measures, plot_f1_score, create_plot_data, calculcate_f1_score



if __name__ == "__main__":
    mode = ['val', 'separate_para', 'overlap_ranks', 'legal_task2_dpr']
    dpr_file = '/mnt/c/Users/salthamm/Documents/coding/DPR/data/coliee2021_task1/{}/aggregate/{}/search_{}_{}_aggregation_{}.txt'.format(
        mode[3], mode[0], mode[0], mode[1], mode[2])
    bm25_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task2/bm25/aggregate/{}/search_val_something_aggregation_overlap_scores.txt'.format(mode[0])
        #'/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bm25/aggregate/{}/separately_para_w_summ_intro/search_{}_separately_para_w_summ_intro_aggregation_{}.txt'.format(
        #mode[0], mode[0], mode[2])
    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task2/bm25_dpr'.format(mode[0])

    dpr_dict = read_in_aggregated_scores(dpr_file, 1000)
    # now remove the document itself from the list of candidates
    dpr_dict2 = {}
    for key, value in dpr_dict.items():
        dpr_dict2.update({key: {}})
        for key2, value2 in value.items():
            if key != key2:
                dpr_dict2.get(key).update({key2:value2})
    dpr_dict = dpr_dict2
    bm25_dict = read_in_aggregated_scores(bm25_file, 1000)

    # check if both dictionaries contain the same query ids
    dpr_keys = list(dpr_dict.keys())
    dpr_keys.sort()
    #assert dpr_keys == list(bm25_dict.keys())

    # read in the label files
    label_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/train/train_wo_val_labels.json'
    label_file_val = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/val/val_labels.json'
    label_file_test = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/test/test_no_labels.json'

    if mode[0] == 'train':
        qrels = read_label_file(label_file)
    elif mode[0] == 'val':
        qrels = read_label_file(label_file_val)
    elif mode[0] == 'test':
        with open(label_file_test, 'rb') as f:
            qrels = json.load(f)
            qrels = [x.split('.txt')[0] for x in qrels]

    #compare_overlap_rel(dpr_dict, bm25_dict, qrels)
    # combine scores could be an idea, also train the combination weights could be an idea! i mean its two weights
    # to combine the scores and then in the end you can apply a list loss!

    # analyze score distribution
    #analyze_score_distribution(dpr_dict, 'dpr', output_dir)
    #analyze_score_distribution(bm25_dict, 'bm25', output_dir)

    measurements = {'recall_1', 'recall_2', 'recall_3', 'recall_4', 'recall_5', 'recall_6', 'recall_7', 'recall_8',
                    'recall_9', 'recall_10','recall_11', 'recall_12', 'recall_13', 'recall_14', 'recall_15', 'recall_16', 'recall_17', 'recall_18',
                    'recall_19', 'recall_20','P_1', 'P_2', 'P_3', 'P_4', 'P_5', 'P_6', 'P_7', 'P_8', 'P_9', 'P_10',
                    'P_11', 'P_12', 'P_13', 'P_14', 'P_15', 'P_16', 'P_17', 'P_18', 'P_19', 'P_20'}

    #measurements = {'recall_1', 'recall_2', 'recall_3', 'recall_4', 'recall_5', 'recall_6', 'recall_7', 'recall_8',
    #                'recall_9', 'recall_10', 'P_1', 'P_2', 'P_3', 'P_4', 'P_5', 'P_6', 'P_7', 'P_8', 'P_9', 'P_10'}

    plotting_data = {}
    weights = [[2, 1], [3, 1], [4, 1], [1,0], [0,1], [1, 1], [1, 2], [1, 3], [1, 4]]
    #weights = [[0,1], [1,0]]
    for weight in weights:
        mode = ['val', 'separate_para_weight_dpr_{}_bm25_{}'.format(weight[0], weight[1]), 'overlap_ranks', 'legal_task2_dpr']
        run, measures = evaluate_weight(dpr_dict, bm25_dict, qrels, mode, weight[0], weight[1], measurements)
        plotting_data.update({'BM25:{} DPR:{}'.format(weight[1], weight[0]): create_plot_data(measures)})
        #output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bm25_dpr/aggregate/{}/'.format(mode[0])
        #sort_write_trec_output(run, output_dir, mode)


    plot_measures(plotting_data, output_dir, 'dpr_bm25_different_weights.svg')
    calculcate_f1_score(plotting_data, output_dir, 'dpr_bm25_different_weights_f1.txt')
    plot_f1_score(plotting_data, output_dir, 'dpr_bm25_different_weights_f1.svg')