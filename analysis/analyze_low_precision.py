import os
import random
import pickle
import pytrec_eval
from eval.eval_bm25_coliee2021 import read_label_file
from analysis.ttest import measure_per_query
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib
from analysis.compare_bm25_dpr import read_in_run_from_pickle, remove_query_from_ranked_list
from analysis.diff_bm25_dpr import first_diff_analysis, write_diff_cases


def ranking_eval(qrels, run, output_dir, measurements, output_file= 'eval_bm25_aggregate_overlap.txt'):
    # trec eval
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measurements)

                                               #{'recall_1', 'recall_2', 'recall_3', 'recall_4', 'recall_5', 'recall_6', 'recall_7', 'recall_8',
                    #'recall_9', 'recall_10','recall_11', 'recall_12', 'recall_13', 'recall_14', 'recall_15', 'recall_16', 'recall_17', 'recall_18',
                    #'recall_19', 'recall_20','P_1', 'P_2', 'P_3', 'P_4', 'P_5', 'P_6', 'P_7', 'P_8', 'P_9', 'P_10',
                    #'P_11', 'P_12', 'P_13', 'P_14', 'P_15', 'P_16', 'P_17', 'P_18', 'P_19', 'P_20'}) # {'recall_100', 'recall_200', 'recall_300', 'recall_500', 'recall_1000'})

    results = evaluator.evaluate(run)

    def print_line(measure, scope, value):
        print('{:25s}{:8s}{:.4f}'.format(measure, scope, value))

    def write_line(measure, scope, value):
        return '{:25s}{:8s}{:.4f}'.format(measure, scope, value)

    per_query = {}
    for query_id, query_measures in sorted(results.items()):
        for measure, value in sorted(query_measures.items()):
            if per_query.get(query_id):
                per_query.get(query_id).update({measure : value})
            else:
                per_query.update({query_id:{}})
                per_query.get(query_id).update({measure: value})
            #print_line(measure, query_id, value)

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

    return per_query


def get_diff_per_query(per_query_baseline, per_query_legalbert_doc, measure):
    diff_per_query = {}
    for query, measurements in per_query_legalbert_doc.items():
        diff_per_query.update({query: (measurements.get(measure) - per_query_baseline.get(query).get(measure))})
    return diff_per_query


def get_performance_per_query(per_query_baseline, measure):
    diff_per_query = {}
    for query, measurements in per_query_baseline.items():
        diff_per_query.update({query: measurements.get(measure)})
    return diff_per_query


def plot_wins_losses(diff_legalbert_doc_r, diff_legalbert_doc_p, measure1, measure2, output_dir, m1_min=-1, m1_max=1, m1_step=1, m2_min=-0.05, m2_max=0.05, m2_step=0.05, color='purple', sort='sort_recall'):
    # now plot them
    x = np.array(list(range(len(diff_legalbert_doc_r.keys()))))
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    # The below code will create two plots. The parameters that .subplot take are (row, column, no. of plots).
    plt.subplot(2, 1, 1)
    #plt.tight_layout()
    # This will create the bar graph for population

    #sns.set_style('darkgrid')
    #sns.set_style("whitegrid")
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    ax = sns.barplot(x, np.array(list(diff_legalbert_doc_r.values())),color=color, edgecolor = color)

    #bar1 = plt.bar(x, diff_legalbert_doc_r.values(),color='purple', edgecolor = 'purple')
    #ax = plt.gca()
    plt.ylim(m1_min, m1_max)
    #ax.set_facecolor('white')
    ax.yaxis.set_ticks(np.arange(m1_min, m1_max+m1_step/2, m1_step))
    plt.ylabel('{}'.format(measure1))
    plt.xticks([], [])
    ax.axhline(y=0, color='black')
    ax.spines['bottom'].set_color('none')
    plt.subplots_adjust(left=0.15) #, right=0.96, top=0.96)

    # The below code will create the second plot.
    plt.subplot(2, 1, 2)
    # This will create the bar graph for gdp i.e gdppercapita divided by population.
    #sns.set_style('darkgrid')
    #sns.set_style("whitegrid")
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    ax = sns.barplot(x, np.array(list(diff_legalbert_doc_p.values())),color=color, edgecolor = color)

    #bar2 = plt.bar(x, diff_legalbert_doc_p.values(),color='purple', edgecolor = 'purple')
    #ax = plt.gca()
    ax.set_ylim([m2_min, m2_max])
    #plt.ylim(-0.05, 0.05)
    ax.yaxis.set_ticks(np.arange(m2_min, m2_max+m2_step/2, m2_step))
    ax.axhline(y=0, color='black')
    ax.spines['bottom'].set_color('none')
    #ax.set_facecolor('white')
    plt.ylabel('{}'.format(measure2))
    plt.xticks([], [])
    if sort=='sort_bm25':
        plt.xlabel('easy -> hard (by BM25 {})'.format(measure1))
    elif sort=='sort_recall':
        plt.xlabel('sorted by {}'.format(measure1))
    else:
        plt.xlabel(sort)
    plt.subplots_adjust(left=0.15)

    plt.savefig(os.path.join(output_dir, '{}_{}_wins_losses_{}.svg'.format(measure1, measure2, sort)))

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()


def sort_diff(diff_legalbert_doc_r, diff_legalbert_doc_p, per_query_baseline, sort):
    if sort == 'sort_bm25':
        sort_baseline = get_performance_per_query(per_query_baseline, measure1)
        sort_baseline = {k: v for k, v in sorted(sort_baseline.items(), key=lambda item: item[1], reverse=True)}
        # now sort by recall of bm25!
        diff_legalbert_doc_r = {k: v for k, v in sorted(diff_legalbert_doc_r.items(),
                                                        key=lambda pair: list(sort_baseline.keys()).index(pair[0]))}
        print(diff_legalbert_doc_r)
        # sort the same way as diff_legalbert_doc
        diff_legalbert_doc_p = {k: v for k, v in sorted(diff_legalbert_doc_p.items(),
                                                        key=lambda pair: list(diff_legalbert_doc_r.keys()).index(pair[0]))}
        print(diff_legalbert_doc_p)
    else:
        # sort by recall increasing order
        diff_legalbert_doc_r = {k: v for k, v in sorted(diff_legalbert_doc_r.items(), key=lambda item: item[1])}
        # sort the same way as diff_legalbert_doc
        diff_legalbert_doc_p = {k: v for k, v in sorted(diff_legalbert_doc_p.items(), key=lambda pair: list(diff_legalbert_doc_r.keys()).index(pair[0]))}
    return diff_legalbert_doc_r, diff_legalbert_doc_p


if __name__ == "__main__":
    # coliee data
    # load bm25 runs
    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bm25/aggregate/test/separately_para_w_summ_intro/'
    with open(os.path.join(output_dir, 'run_bm25_aggregated_test_whole_doc_overlap_docs.pickle'), 'rb') as f:
       run_bm25_doc = pickle.load(f)
    with open(os.path.join(output_dir, 'run_aggregated_test_rrf_overlap_ranks.pickle'), 'rb') as f:
       run_bm25_parm = pickle.load(f)

    # load bert para based runs
    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/dpr/legal_task2/bert/aggregate/test/'
    with open(os.path.join(output_dir, 'run_aggregated_test_separate_para_rrf.pickle'), 'rb') as f:
       run_bert_parm_rrf = pickle.load(f)
    with open(os.path.join(output_dir, 'run_aggregated_test_vrrf.pickle'), 'rb') as f:
       run_bert_parm_vrrf = pickle.load(f)

    # load legalbert para based runs
    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/dpr/legal_task2/legalbert/aggregate/test/'
    with open(os.path.join(output_dir, 'run_aggregated_test_rrf_overlap_ranks.pickle'), 'rb') as f:
       run_legbert_para_rrf = pickle.load(f)
    with open(os.path.join(output_dir, 'run_aggregated_test_vrrf.pickle'), 'rb') as f:
       run_legbert_para_vrrf = pickle.load(f)

    # load legalbert doc based runs
    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/dpr/legal_task1/legalbert/eval/test/'
    with open(os.path.join(output_dir, 'run_aggregated_test_separate_para_rrf.pickle'), 'rb') as f:
       run_legbert_doc_rrf = pickle.load(f)
    with open(os.path.join(output_dir, 'run_aggregated_test_vrrf_legalbert_doc.pickle'), 'rb') as f:
       run_legbert_doc_vrrf = pickle.load(f)

    qrels = read_label_file(
        '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/test/task1_test_labels_2021.json')

    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/analysis_low_precision'

    # 1. evaluate ndcg@100/500/1k
    measurements = {'recall_100', 'recall_200', 'recall_300', 'recall_500', 'recall_1000', 'ndcg_cut_10', 'ndcg_cut_100', 'ndcg_cut_500', 'ndcg_cut_1000', 'P_10', 'P_100', 'P_500', 'P_1000'}
    #measurements = {'recall_1', 'recall_2', 'recall_3', 'recall_4', 'recall_5', 'recall_6', 'recall_7', 'recall_8',
                    #'recall_9', 'recall_10','recall_11', 'recall_12', 'recall_13', 'recall_14', 'recall_15', 'recall_16', 'recall_17', 'recall_18',
                    #'recall_19', 'recall_20','P_1', 'P_2', 'P_3', 'P_4', 'P_5', 'P_6', 'P_7', 'P_8', 'P_9', 'P_10',
                    #'P_11', 'P_12', 'P_13', 'P_14', 'P_15', 'P_16', 'P_17', 'P_18', 'P_19', 'P_20'}
    per_query = ranking_eval(qrels, run_bm25_parm, output_dir, measurements, 'eval_bm25_parm_ncdg.txt')

    # 2. compare recall, precision @500 per query, and then also the delta r, delta p to baseline (bm25 doc) per query
    # is there a correlation

    per_query_baseline = ranking_eval(qrels, run_bm25_doc, output_dir, measurements, 'eval_bm25_whole_doc_ncdg.txt')
    per_query_legalbert_doc = ranking_eval(qrels, run_legbert_doc_vrrf, output_dir, measurements,
                                           'eval_legalbert_doc_vrrf_ncdg.txt')
    per_query_legalbert_para = ranking_eval(qrels, run_legbert_para_vrrf, output_dir, measurements,
                                           'eval_legalbert_para_vrrf_ncdg.txt')
    per_query_bm25_parm = ranking_eval(qrels, run_bm25_parm, output_dir, measurements, 'eval_bm25_parm_ncdg.txt')

    # use for comparion of bm25 doc and bm25 parm rrf
    #per_query_legalbert_doc = per_query_bm25_parm

    n = 1000
    sort = 'sort_recall'

    # difference plots

    measure1 = 'recall_{}'.format(n)
    diff_legalbert_doc_r = get_diff_per_query(per_query_baseline, per_query_legalbert_doc, measure1)
    measure2 = 'P_{}'.format(n)
    diff_legalbert_doc_p = get_diff_per_query(per_query_baseline, per_query_legalbert_doc, measure2)

    diff_legalbert_doc_r, diff_legalbert_doc_p = sort_diff(diff_legalbert_doc_r, diff_legalbert_doc_p, per_query_baseline, sort)

    if n==100:
        color = 'violet'
    elif n==500:
        color = 'purple'
    elif n==1000:
        color = 'indigo'

    plot_wins_losses(diff_legalbert_doc_r, diff_legalbert_doc_p, measure1, measure2, output_dir, -1, 1, 1, -0.05, 0.05, 0.05, color, sort)


    # now plots where you have the bm25 performance vs legalbertdoc performance in terms of 1 measure
    n = 100
    one_measure = 'P_{}'.format(n)
    if n==100:
        color = 'violet'
    elif n==500:
        color = 'purple'
    elif n==1000:
        color = 'indigo'
    elif n==10:
        color = 'pink'

    # get the performance
    per_query_baseline_measure = get_performance_per_query(per_query_baseline, one_measure)
    per_query_legalbert_doc_measure = get_performance_per_query(per_query_legalbert_doc, one_measure)

    # sorted after baseline
    per_query_baseline_measure, per_query_legalbert_doc_measure = sort_diff(per_query_baseline_measure, per_query_legalbert_doc_measure, per_query_baseline, 'sort_recall')

    plot_wins_losses(per_query_baseline_measure, per_query_legalbert_doc_measure, 'BM25', 'LegalBERT Doc', output_dir, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5,
                     color, one_measure)

    #
    # now qualitative analysis of the biggest wins and losses of the cases, write the cases in an ouptut folder
    #

    dpr_dict_doc = remove_query_from_ranked_list(run_legbert_doc_vrrf)
    bm25_dict_doc = remove_query_from_ranked_list(run_bm25_doc)

    # legalbert doc vs bm25 doc
    dpr_dict_doc_rel, bm25_dict_doc_rel, query_diff_doc, query_diff_length_doc = first_diff_analysis(dpr_dict_doc,
                                                                                                     bm25_dict_doc,
                                                                                                     qrels)

    # then i only want to write the cases where the wins/losses are high!

    corpus_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/corpus'
    pickle_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/pickle_files'
    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/analysis_low_precision/qual/output/legalbert_doc'

    write_diff_cases(query_diff_length_doc, query_diff_doc, pickle_dir, output_dir, corpus_dir)

    # table: per query: recall, precision@n and delta recall and delta precision@n, analyze in pandas dataframe?