import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from eval.eval_bm25_coliee2021 import read_label_file #ranking_eval
from retrieval.bm25_aggregate_paragraphs import sort_write_trec_output
from preprocessing.coliee21_task2_bm25 import ranking_eval
from preprocessing.caselaw_stat_corpus import preprocess_label_file


def plot_measures(measures, eval_dir, plot_file):
    plt.figure(figsize=(10, 8))
    plt.xlabel('recall', fontsize=15)
    plt.ylabel('precision', fontsize=15)
    for name, measure in measures.items():
        xs, ys = zip(*measure.values())
        labels = measure.keys()
        # display
        plt.scatter(xs, ys, marker='o')
        plt.plot(xs, ys, label=name)
        for label, x, y in zip(labels, xs, ys):
            plt.annotate(label, xy=(x, y))
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(eval_dir, plot_file))


def calculcate_f1_score(plotting_data, output_dir, output_file):
    with open(os.path.join(output_dir, output_file), 'w') as f:
        for name, measure in plotting_data.items():
            for key, value in measure.items():
                f1_score = 2*value[0]*value[1]/(value[0]+value[1])
                f.writelines(' '.join([name, key, str(f1_score)]) + '\n')


def plot_f1_score(plotting_data, eval_dir, plot_file):
    f1_dict = {}
    for name, measure in plotting_data.items():
        f1_dict.update({name: {}})
        for key, value in measure.items():
            f1_score = 2 * value[0] * value[1] / (value[0] + value[1])
            f1_dict.get(name).update({key: [key, f1_score]})
    plt.figure(figsize=(10, 8))
    plt.xlabel('cut-off value', fontsize=15)
    plt.ylabel('f1-score', fontsize=15)
    for name, measure in f1_dict.items():
        xs, ys = zip(*measure.values())
        labels = measure.keys()
        plt.scatter(xs, ys, marker='o')
        plt.plot(xs, ys, label=name)
        for label, x, y in zip(labels, xs, ys):
            plt.annotate(label, xy=(x, y))
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(eval_dir, plot_file))


def create_plot_data(measures):
    plotting_data = {}
    for key, value in measures.items():
        if not plotting_data.get(key.split('_')[1]):
            plotting_data.update({key.split('_')[1]: [0, 0]})
        if 'P' in key:
            plotting_data.get(key.split('_')[1])[1] = value
        if 'recall' in key:
            plotting_data.get(key.split('_')[1])[0] = value

    # order them:
    desired_order_list = [int(key) for key, value in plotting_data.items()]
    desired_order_list.sort()
    desired_order_list = [str(x) for x in desired_order_list]
    plotting_data_sorted = {k: plotting_data[k] for k in desired_order_list}
    return plotting_data_sorted


def create_plot_data_recall(measures):
    plotting_data = {}
    for key, value in measures.items():
        if 'recall' in key:
            plotting_data.update({key: [key.split('_')[1],value]})

    # order them:
    desired_order_list = ['recall_100', 'recall_200', 'recall_300', 'recall_500', 'recall_1000']
    plotting_data_sorted = {k: plotting_data[k] for k in desired_order_list}
    return plotting_data_sorted


def plot_recall(measures, eval_dir, plot_file):
    plt.figure(figsize=(10, 8))
    plt.xlabel('cut-off', fontsize=15)
    plt.ylabel('recall', fontsize=15)
    for name, measure in measures.items():
        xs, ys = zip(*measure.values())
        labels = measure.keys()
        # display
        plt.scatter(xs, ys, marker='o')
        plt.plot(xs, ys, label=name)
        #for label, x, y in zip(labels, xs, ys):
        #    plt.annotate(label, xy=(x, y))
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(eval_dir, plot_file))


def read_in_aggregated_scores(input_file: str, top_n=1000):
    with open(input_file) as fp:
        lines = fp.readlines()
        file_dict = {}
        for line in lines:
            line_list = line.strip().split(' ')
            assert len(line_list) == 6
            if int(line_list[3]) <= top_n:
                if file_dict.get(line_list[0]):
                    file_dict.get(line_list[0]).update({line_list[2]: [int(line_list[3]), float(line_list[4])]})
                else:
                    file_dict.update({line_list[0]: {}})
                    file_dict.get(line_list[0]).update({line_list[2]: [int(line_list[3]), float(line_list[4])]})
    return file_dict


def return_rel_docs_for_dict(labels: dict, dpr_dict: dict):
    # filter the dictionary for only the relevant ones
    label_keys = [key for key in labels]
    dpr_keys = [key for key in dpr_dict]
    assert label_keys.sort() == dpr_keys.sort()

    print('im here')

    filtered_dict = {}
    for query_id in labels.keys():
        filtered_dict.update({query_id: {}})
        rel_docs = labels.get(query_id)
        for doc in rel_docs:
            if dpr_dict.get(query_id).get(doc):
                filtered_dict.get(query_id).update({doc: dpr_dict.get(query_id).get(doc)})

    return filtered_dict


def compare_overlap(dpr_dict_rel, bm25_dict_rel):
    intersection_bm25 = []
    intersection_dpr = []
    for query_id in dpr_dict_rel.keys():
        dpr_rel_doc = set(dpr_dict_rel.get(query_id).keys())
        bm25_rel_doc = set(bm25_dict_rel.get(query_id).keys())
        print(query_id)
        if bm25_rel_doc:
            intersection_bm25.append(len(dpr_rel_doc.intersection(bm25_rel_doc)) / len(bm25_rel_doc))
        # else:
        #    intersection_bm25.append(0)
        if dpr_rel_doc:
            intersection_dpr.append(len(dpr_rel_doc.intersection(bm25_rel_doc)) / len(dpr_rel_doc))
        # else:
        #    intersection_dpr.append(0)

    print('average percentual intersection of bm25 results which are also found in dpr {}'.format(
        np.mean(intersection_bm25)))
    print('average percentual intersection of dpr results which are also found in bm25 {}'.format(
        np.mean(intersection_dpr)))


def analyze_score_distribution(dpr_dict, name, output_dir):
    scores = []
    for query in dpr_dict.keys():
        for doc in dpr_dict.get(query).keys():
            scores.append(dpr_dict.get(query).get(doc)[1])

    print('maximum score is {}'.format(max(scores)))
    print('minimum score is {}'.format(min(scores)))

    # plot histogram
    #sns.distplot(scores, kde=False, color='red')
    plt.xlim(0, 200)
    plt.hist(scores, bins=1000)
    plt.xlabel('scores', fontsize=16)
    plt.savefig(os.path.join(output_dir, 'scores_{}.svg'.format(name)))
    plt.clf()

    # boxplot
    sns.boxplot(y=scores)
    plt.xlabel('scores', fontsize=16)
    plt.savefig(os.path.join(output_dir, 'scores_{}2.svg'.format(name)))
    plt.clf()


def evaluate_weighting(dpr_dict, bm25_dict, qrels, output_dir, output_file, weight_dpr, weight_bm25, measurements):
    # aggregate scores with certain weight
    run = {}
    for query_id in dpr_dict.keys():
        run.update({query_id: {}})
        for doc in dpr_dict.get(query_id).keys():
            run.get(query_id).update({doc: weight_dpr * dpr_dict.get(query_id).get(doc)[1]})
        for doc in bm25_dict.get(query_id).keys():
            if run.get(query_id).get(doc):
                run.get(query_id).update({doc: run.get(query_id).get(doc) + weight_bm25 * bm25_dict.get(query_id).get(doc)[1]})
            else:
                run.get(query_id).update({doc: weight_bm25 * bm25_dict.get(query_id).get(doc)[1]})

    # evaluate aggregated list!
    measures = ranking_eval(qrels, run, output_dir, output_file, measurements)
    #measures = None

    return run, measures


def compare_overlap_rel(dpr_dict, bm25_dict, qrels):
    # filter the dictionary for only the relevant ones
    #dpr_dict_rel = return_rel_docs_for_dict(qrels, dpr_dict)
    #bm25_dict_rel = return_rel_docs_for_dict(qrels, bm25_dict)

    # now compare overlap of relevant docs
    compare_overlap(dpr_dict, bm25_dict)


def evaluate_weight(dpr_dict, bm25_dict, qrels, mode, weight_dpr, weight_bm25, measurements=
{'recall_1', 'recall_2', 'recall_3', 'recall_4', 'recall_5', 'recall_6', 'recall_7', 'recall_8', 'recall_9', 'recall_10',
                                                       'P_1', 'P_2', 'P_3', 'P_4', 'P_5', 'P_6', 'P_7', 'P_8', 'P_9', 'P_10'}):
    output_dir2 = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/bm25_dpr'
    output_file = 'eval_score_{}_{}_weight_dpr_{}_weight_bm25_{}.txt'.format(mode[0], mode[2], weight_dpr, weight_bm25)

    run, measures = evaluate_weighting(dpr_dict, bm25_dict, qrels, output_dir2, output_file, weight_dpr, weight_bm25, measurements)

    return run, measures


def remove_query_from_ranked_list(dpr_dict):
    # now remove the document itself from the list of candidates
    dpr_dict2 = {}
    for key, value in dpr_dict.items():
        dpr_dict2.update({key: {}})
        for key2, value2 in value.items():
            if key != key2:
                dpr_dict2.get(key).update({key2: value2})
    return dpr_dict2


def eval_prec_rec(measurements, weights, mode, dpr_dict, bm25_dict, qrels, output_dir):
    plotting_data = {}
    for weight in weights:
        mode = [mode[0], 'separate_para_weight_dpr_{}_bm25_{}'.format(weight[0], weight[1]), mode[2], mode[3]]
        run, measures = evaluate_weight(dpr_dict, bm25_dict, qrels, mode, weight[0], weight[1], measurements)
        plotting_data.update({'BM25:{} DPR:{}'.format(weight[1], weight[0]): create_plot_data(measures)})
        output_dir2 = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bm25_dpr_legalbert/aggregate/{}/'.format(mode[0])
        sort_write_trec_output(run, output_dir2, mode)

    plot_measures(plotting_data, output_dir, 'dpr_bm25_different_weights_test2.svg')
    calculcate_f1_score(plotting_data, output_dir, 'dpr_bm25_different_weights_f15_test.txt')
    plot_f1_score(plotting_data, output_dir, 'dpr_bm25_different_weights_f15_test.svg')

    return plotting_data


def read_in_run_from_pickle(bm25_file):

    with open(bm25_file, 'rb') as f:
        bm25_dict = pickle.load(f)

    bm25_dict_new = {}
    for key, value in bm25_dict.items():
        bm25_dict_new.update({key:{}})
        i = 1
        for key2, value2 in value.items():
            bm25_dict_new.get(key).update({key2: [i, float(value2)]})
            i += 1

    return bm25_dict_new


if __name__ == "__main__":
    mode = ['train', 'separate_para', 'rrf', 'legal_task2']
    dpr_file = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/dpr/legalbert_doc/eval/run_dpr_aggregate_rrf.pickle'  #for rrf, vrrf, firstp
    bm25_file = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/bm25/eval/run_bm25_aggregate2_rrf_overlap_ranks.pickle' # change for firstp to whole doc
    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/bm25_dpr/'

    dpr_dict = read_in_run_from_pickle(dpr_file)
    dpr_dict = remove_query_from_ranked_list(dpr_dict)

    bm25_dict = read_in_run_from_pickle(bm25_file)
    bm25_dict = remove_query_from_ranked_list(bm25_dict)

    bm25_dict_new = {}
    for key, value in bm25_dict.items():
        bm25_dict_new.update({'id{}'.format(key):{}})
        for key2, value2 in value.items():
            bm25_dict_new.get('id{}'.format(key)).update({'id{}'.format(key2):value2})

    bm25_dict = bm25_dict_new

    # read in the label files
    label_file = '/mnt/c/Users/salthamm/Documents/coding/ussc-caselaw-collection/airs2017-collection/qrel.txt'

    qrels = preprocess_label_file(label_file)
    qrels_updated = {}
    for key, value in qrels.items():
        qrels_updated.update({key: {}})
        for val in value:
            qrels_updated.get(str(key)).update({str(val): 1})

    qrels_dict_new = {}
    for key, value in qrels_updated.items():
        qrels_dict_new.update({'id{}'.format(key): {}})
        for key2, value2 in value.items():
            qrels_dict_new.get('id{}'.format(key)).update({'id{}'.format(key2): value2})

    qrels_updated = qrels_dict_new

    #compare_overlap_rel(dpr_dict, bm25_dict, qrels_updated)
    #analyze_score_distribution(dpr_dict, 'dpr', output_dir)
    #analyze_score_distribution(bm25_dict, 'bm25', output_dir)

    measurements = {'recall_100', 'recall_200', 'recall_300', 'recall_500', 'recall_1000', 'ndcg_cut_10'}
    weights = [[2, 1], [3, 1], [4, 1], [1,0], [0,1], [1, 1], [1, 2], [1, 3], [1, 4]]

    plotting_data = {}
    plotting_data2 = {}
    for weight in weights:
        mode = ['train', 'separate_para_weight_dpr_{}_bm25_{}'.format(weight[0], weight[1]), 'rrf',
                'legal_task2']
        run, measures = evaluate_weight(dpr_dict, bm25_dict, qrels_updated, mode, weight[0], weight[1], measurements)
        #if 'recall_100' in measurements:
        #    plotting_data.update({'BM25:{} DPR:{}'.format(weight[1], weight[0]): create_plot_data_recall(measures)})
        #else:
        #    plotting_data.update({'BM25:{} DPR:{}'.format(weight[1], weight[0]): create_plot_data(measures)})
        plotting_data2.update({'BM25:{} DPR:{}'.format(weight[1], weight[0]): measures})

    #print(plotting_data)
    print(plotting_data2)

    #plotting_data.update({'DPR': plotting_data.get('BM25:0 DPR:1')})
    #plotting_data.update({'BM25': plotting_data.get('BM25:1 DPR:0')})

    #plotting_data.pop('BM25:1 DPR:0')
    #plotting_data.pop('BM25:0 DPR:1')

    #plot_recall(plotting_data, output_dir, 'dpr_bm25_different_weights_{}_cutoff5_all_bert_doc_dpr.svg'.format(mode[0]))
    #if 'recall_1' in measurements:
    #    plot_f1_score(plotting_data, output_dir, 'dpr_bm25_different_weights_f1_score_{}_top20.svg'.format(mode[0]))
    #    calculcate_f1_score(plotting_data, output_dir, 'f1-scores_dpr_bm25_{}_top20.txt'.format(mode[0]))

    # here i should write the plotting data
    print(plotting_data2)

    with open(os.path.join(output_dir, 'plotting_data_weights_recall_precision_top_20.txt'), 'w') as f:
        for key, value in plotting_data2.items():
            for key2, value2 in value.items():
                f.writelines('{}\t{}\t{}\n'.format(key, key2, value2[1]))