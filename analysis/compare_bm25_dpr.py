import os
import json
import pickle
import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from eval.eval_bm25_coliee2021 import read_label_file #ranking_eval
from retrieval.bm25_aggregate_paragraphs import sort_write_trec_output
from preprocessing.coliee21_task2_bm25 import ranking_eval
from analysis.ttest import measure_per_query


def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None):
    """Return an axes of confidence bands using a simple approach.

    Notes
    -----
    .. math:: \left| \: \hat{\mu}_{y|x0} - \mu_{y|x0} \: \right| \; \leq \; T_{n-2}^{.975} \; \hat{\sigma} \; \sqrt{\frac{1}{n}+\frac{(x_0-\bar{x})^2}{\sum_{i=1}^n{(x_i-\bar{x})^2}}}
    .. math:: \hat{\sigma} = \sqrt{\sum_{i=1}^n{\frac{(y_i-\hat{y})^2}{n-2}}}

    References
    ----------
    .. [1] M. Duarte.  "Curve fitting," Jupyter Notebook.
       http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/CurveFitting.ipynb

    """
    if ax is None:
        ax = plt.gca()

    ci = t * s_err * np.sqrt(1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    ax.fill_between(x2, y2 + ci, y2 - ci, color="#b9cfe7", edgecolor="")

    return ax


def plot_ci_bootstrap(xs, ys, resid, nboot=500, ax=None):
    """Return an axes of confidence bands using a bootstrap approach.

    Notes
    -----
    The bootstrap approach iteratively resampling residuals.
    It plots `nboot` number of straight lines and outlines the shape of a band.
    The density of overlapping lines indicates improved confidence.

    Returns
    -------
    ax : axes
        - Cluster of lines
        - Upper and Lower bounds (high and low) (optional)  Note: sensitive to outliers

    References
    ----------
    .. [1] J. Stults. "Visualizing Confidence Intervals", Various Consequences.
       http://www.variousconsequences.com/2010/02/visualizing-confidence-intervals.html

    """
    if ax is None:
        ax = plt.gca()

    bootindex = sp.random.randint

    for _ in range(nboot):
        resamp_resid = resid[bootindex(0, len(resid) - 1, len(resid))]
        # Make coeffs of for polys
        pc = sp.polyfit(xs, ys + resamp_resid, 1)
        # Plot bootstrap cluster
        ax.plot(xs, sp.polyval(pc, xs), "b-", linewidth=2, alpha=3.0 / float(nboot))

    return ax


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
    desired_order_list = ['recall_100','recall_150','recall_200','recall_250','recall_300','recall_350','recall_400','recall_450',
                    'recall_500','recall_550','recall_600','recall_650','recall_700','recall_750','recall_800','recall_850','recall_900','recall_950','recall_1000']
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


def plot_recall_nice(plotting_data, output_dir, plot_file):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    plt.xlabel('cut-off', fontsize=15)
    plt.grid(True, linewidth=0.5, color='lightgrey', linestyle='-')
    plt.ylabel('recall', fontsize=15)
    for name, measure in plotting_data.items():
        xs, ys = zip(*measure.values())
        labels = measure.keys()
        # display
        plt.scatter(xs, ys, marker='o')
        plt.plot(xs, ys, label=name)
        # for label, x, y in zip(labels, xs, ys):
        #    plt.annotate(label, xy=(x, y))
    plt.legend(loc="lower right")
    plt.xticks(range(1,21,2),range(100,1100,100))
    ax.patch.set_facecolor('white')
    plt.savefig(os.path.join(output_dir, plot_file), bbox_inches = 'tight')


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
    label_keys = list(labels.keys())
    dpr_keys = list(dpr_dict.keys())
    assert label_keys.sort() == dpr_keys.sort()

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

    return run, measures


def compare_overlap_rel(dpr_dict, bm25_dict, qrels):
    # filter the dictionary for only the relevant ones
    dpr_dict_rel = return_rel_docs_for_dict(qrels, dpr_dict)
    bm25_dict_rel = return_rel_docs_for_dict(qrels, bm25_dict)

    # now compare overlap of relevant docs
    compare_overlap(dpr_dict_rel, bm25_dict_rel)


def evaluate_weight(dpr_dict, bm25_dict, qrels, mode, weight_dpr, weight_bm25, measurements=
{'recall_1', 'recall_2', 'recall_3', 'recall_4', 'recall_5', 'recall_6', 'recall_7', 'recall_8', 'recall_9', 'recall_10',
                                                       'P_1', 'P_2', 'P_3', 'P_4', 'P_5', 'P_6', 'P_7', 'P_8', 'P_9', 'P_10'}):
    output_dir2 = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bm25_dpr_legalbert/eval/{}'.format(mode[0])
    output_file = 'eval22_score_{}_{}_weight_dpr_{}_weight_bm25_{}.txt'.format(mode[0], mode[2], weight_dpr, weight_bm25)

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


def compute_ci(run1, qrels, measurement, conf_int=0.95):
    # standard deviation
    run1_list = measure_per_query(run1, qrels, measurement)
    mean = np.mean(run1_list)
    std_dev = np.std(run1_list)
    #print(stats.norm.interval(conf_int, loc=mean, scale=std_dev))
    ci = stats.norm.interval(conf_int, loc=mean, scale=std_dev)
    return ci


def create_ci_data(run, qrels, measurements):
    cis = {}
    for measure in measurements:
        cis.update({measure: compute_ci(run, qrels, measure)})
    desired_order_list = ['recall_100', 'recall_150', 'recall_200', 'recall_250', 'recall_300', 'recall_350',
                          'recall_400', 'recall_450',
                          'recall_500', 'recall_550', 'recall_600', 'recall_650', 'recall_700', 'recall_750',
                          'recall_800', 'recall_850', 'recall_900', 'recall_950', 'recall_1000']
    plotting_data_sorted = {k: cis[k] for k in desired_order_list}
    return plotting_data_sorted


if __name__ == "__main__":
    mode = ['test', 'separate_para', 'rrf', 'legal_task2']
    dpr_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/dpr/{}/legalbert/aggregate/{}/run_aggregated_{}_{}.pickle'.format(
         mode[3], mode[0], mode[0], mode[2])
    # #'/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/dpr/{}/legalbert/aggregate/{}/search_{}_{}_aggregation_{}.txt'.format(
    # #    mode[3], mode[0], mode[0], mode[1], mode[2])
    # bm25_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bm25/aggregate/{}/separately_para_w_summ_intro/run_aggregated_{}_{}.pickle'.format(
    #     mode[0], mode[0], mode[2])
    # bm25_file2= '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bm25/aggregate/{}/separately_para_w_summ_intro/search_{}_separately_para_w_summ_intro_aggregation_{}.txt'.format(
    #     mode[0], mode[0], mode[2])
    # output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bm25_dpr_legalbert/plot/{}'.format(mode[0])
    #
    # dpr_dict = read_in_run_from_pickle(dpr_file)
    # dpr_dict = remove_query_from_ranked_list(dpr_dict)
    #
    # bm25_dict = read_in_run_from_pickle(bm25_file)
    # bm25_dict = remove_query_from_ranked_list(bm25_dict)
    #
    # # read in the label files
    # label_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/train/train_wo_val_labels.json'
    # label_file_val = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/val/val_labels.json'
    # label_file_test = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/test/task1_test_labels_2021.json'
    #
    # if mode[0] == 'train':
    #     qrels = read_label_file(label_file)
    # elif mode[0] == 'val':
    #     qrels = read_label_file(label_file_val)
    # elif mode[0] == 'test':
    #     qrels = read_label_file(label_file_test)
    #     #with open(label_file_test, 'rb') as f:
    #     #    qrels = json.load(f)
    #     #    qrels = [x.split('.txt')[0] for x in qrels]
    #
    # compare_overlap_rel(dpr_dict, bm25_dict, qrels)
    # analyze_score_distribution(dpr_dict, 'dpr', output_dir)
    # analyze_score_distribution(bm25_dict, 'bm25', output_dir)
    #
    # #measurements = {'recall_1', 'recall_2', 'recall_3', 'recall_4', 'recall_5', 'recall_6', 'recall_7', 'recall_8',
    # #                'recall_9', 'recall_10','recall_11', 'recall_12', 'recall_13', 'recall_14', 'recall_15', 'recall_16', 'recall_17', 'recall_18',
    # #                'recall_19', 'recall_20','P_1', 'P_2', 'P_3', 'P_4', 'P_5', 'P_6', 'P_7', 'P_8', 'P_9', 'P_10',
    # #                'P_11', 'P_12', 'P_13', 'P_14', 'P_15', 'P_16', 'P_17', 'P_18', 'P_19', 'P_20'}
    #
    # measurements = {'recall_50','recall_100','recall_150','recall_200','recall_250','recall_300','recall_350','recall_400','recall_450',
    #                 'recall_500','recall_550','recall_600','recall_650','recall_700','recall_750','recall_800','recall_850','recall_900','recall_950','recall_1000'}
    # weights = [[1,0], [0,1], [1, 1]] #[[2, 1], [3, 1], [4, 1], [1,0], [0,1], [1, 1], [1, 2], [1, 3], [1, 4]]
    #
    # #weights = [[2, 1], [3, 1], [4, 1], [1,0], [0,1], [1, 1], [1, 2], [1, 3], [1, 4]]
    #
    # plotting_data = {}
    # plotting_data2 = {}
    # for weight in weights:
    #     mode = ['test', 'separate_para_weight_dpr_{}_bm25_{}'.format(weight[0], weight[1]), 'overlap_ranks',
    #             'legal_task2']
    #     run, measures = evaluate_weight(dpr_dict, bm25_dict, qrels, mode, weight[0], weight[1], measurements)
    #     if 'recall_100' in measurements:
    #         plotting_data.update({'BM25:{} DPR:{}'.format(weight[1], weight[0]): create_plot_data_recall(measures)})
    #     else:
    #         plotting_data.update({'BM25:{} DPR:{}'.format(weight[1], weight[0]): create_plot_data(measures)})
    #     plotting_data2.update({'BM25:{} DPR:{}'.format(weight[1], weight[0]): measures})
    #
    # plotting_data.update({'BM25+DPR': plotting_data.get('BM25:1 DPR:1')})
    # plotting_data.update({'DPR': plotting_data.get('BM25:0 DPR:1')})
    # plotting_data.update({'BM25': plotting_data.get('BM25:1 DPR:0')})
    #
    # plotting_data.pop('BM25:1 DPR:0')
    # plotting_data.pop('BM25:0 DPR:1')
    # plotting_data.pop('BM25:1 DPR:1')
    #
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111)
    # plt.xlabel('Cut-off', fontsize=15)
    # plt.grid(True, linewidth=0.5, color='lightgrey', linestyle='-')
    # plt.ylabel('Recall', fontsize=15)
    # i = 0
    # colours = ['#d88144','#7fa955', '#5670bc']
    # for name, measure in plotting_data.items():
    #     xs, ys = zip(*measure.values())
    #     labels = measure.keys()
    #     # display
    #     plt.scatter(xs, ys, c= colours[i], marker='o', edgecolors=colours[i])
    #     plt.plot(xs, ys, colours[i], label=name)
    #     # for label, x, y in zip(labels, xs, ys):
    #     #    plt.annotate(label, xy=(x, y))
    #     i +=1
    # plt.legend(loc="lower right")
    # plt.xticks(range(1, 21, 2), range(100, 1100, 100))
    # ax.patch.set_facecolor('white')
    # plt.savefig(os.path.join(output_dir, 'dpr_bm25_different_weights_{}_morerec8_dpr.svg'.format(mode[0])), bbox_inches='tight')
    #
    # plot_recall(plotting_data, output_dir, 'dpr_bm25_different_weights_{}_morerec2_dpr.svg'.format(mode[0]))
    # if 'recall_1' in measurements:
    #     plot_f1_score(plotting_data, output_dir, 'dpr_bm25_different_weights_f1_score_{}_top20.svg'.format(mode[0]))
    #     calculcate_f1_score(plotting_data, output_dir, 'f1-scores_dpr_bm25_{}_top20.txt'.format(mode[0]))
    #
    # # here i should write the plotting data
    # print(plotting_data2)
    #
    # with open(os.path.join(output_dir, 'plotting_data_weights_recall_precision_top_20.txt'), 'w') as f:
    #     for key, value in plotting_data2.items():
    #         for key2, value2 in value.items():
    #             f.writelines('{}\t{}\t{}\n'.format(key, key2, value2[1]))

    mode = ['test', 'separate_para', 'rrf', 'legal_task2']
    dpr_file_doc = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/dpr/{}/legalbert/aggregate/{}/run_dpr_aggregated_{}_whole_doc_overlap_docs.pickle'.format(
        mode[3], mode[0], mode[0])
    #dpr_file_parm = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/dpr/{}/legalbert/aggregate/{}/run_aggregated_test_rrf_overlap_ranks.pickle'.format(
    #         mode[3], mode[0])
    dpr_file_parm = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/dpr/legal_task1/legalbert/eval/test/run_aggregated_test_vrrf_legalbert_doc.pickle'
    #dpr_file_parm = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/dpr/{}/legalbert/aggregate/{}/run_dpr_aggregated_{}_parm_overlap_ranks.pickle'.format(
    #    mode[3], mode[0], mode[0])
    # '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/dpr/{}/legalbert/aggregate/{}/search_{}_{}_aggregation_{}.txt'.format(
    #    mode[3], mode[0], mode[0], mode[1], mode[2])
    bm25_file_doc = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bm25/aggregate/{}/separately_para_w_summ_intro/run_bm25_aggregated_{}_whole_doc_overlap_docs.pickle'.format(
        mode[0], mode[0])
    #bm25_file_parm = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bm25/aggregate/{}/separately_para_w_summ_intro/run_bm25_aggregated_{}_parm_{}.pickle'.format(
    #    mode[0], mode[0], mode[2])
    bm25_file_parm = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bm25/aggregate/test/separately_para_w_summ_intro/run_aggregated_test_rrf_overlap_ranks.pickle'
    #run_aggregated_{}_{}.pickle'.format(
    #         mode[0], mode[0], mode[2])

    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bm25_dpr_legalbert/plot/{}'.format(mode[0])

    dpr_dict_doc = read_in_run_from_pickle(dpr_file_doc)
    dpr_dict_doc = remove_query_from_ranked_list(dpr_dict_doc)

    bm25_dict_doc = read_in_run_from_pickle(bm25_file_doc)
    bm25_dict_doc = remove_query_from_ranked_list(bm25_dict_doc)

    # read in the label files
    label_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/train/train_wo_val_labels.json'
    label_file_val = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/val/val_labels.json'
    label_file_test = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/test/task1_test_labels_2021.json'

    if mode[0] == 'train':
        qrels = read_label_file(label_file)
    elif mode[0] == 'val':
        qrels = read_label_file(label_file_val)
    elif mode[0] == 'test':
        qrels = read_label_file(label_file_test)

    # compare_overlap_rel(dpr_dict_doc, bm25_dict_doc, qrels)
    #
    measurements = {'recall_100', 'recall_150', 'recall_200', 'recall_250', 'recall_300', 'recall_350',
                      'recall_400', 'recall_450',
                      'recall_500', 'recall_550', 'recall_600', 'recall_650', 'recall_700', 'recall_750', 'recall_800',
                      'recall_850', 'recall_900', 'recall_950', 'recall_1000'}
    weights = [[1,0], [0,1], [1, 1]]

    plotting_data = {}
    plotting_data2 = {}
    confidence_intervals = {}
    for weight in weights:
         mode = ['test', 'separate_para_weight_dpr_{}_bm25_{}'.format(weight[0], weight[1]), 'overlap_ranks',
                 'legal_task2']
         run, measures = evaluate_weight(dpr_dict_doc, bm25_dict_doc, qrels, mode, weight[0], weight[1], measurements)
         if 'recall_100' in measurements:
             plotting_data.update({'BM25:{} DPR:{}'.format(weight[1], weight[0]): create_plot_data_recall(measures)})
         else:
             plotting_data.update({'BM25:{} DPR:{}'.format(weight[1], weight[0]): create_plot_data(measures)})
         confidence_intervals.update({'BM25:{} DPR:{}'.format(weight[1], weight[0]): create_ci_data(run, qrels, measurements)})
         plotting_data2.update({'BM25:{} DPR:{}'.format(weight[1], weight[0]): measures})

    #print(plotting_data2)
    print(plotting_data)

    #plotting_data.update({'BM25+DPR': plotting_data.get('BM25:1 DPR:1')})
    plotting_data.update({'Doc FirstP (DPR)': plotting_data.get('BM25:0 DPR:1')})
    plotting_data.update({'Doc (BM25)': plotting_data.get('BM25:1 DPR:0')})
    plotting_data.pop('BM25:1 DPR:0')
    plotting_data.pop('BM25:0 DPR:1')
    plotting_data.pop('BM25:1 DPR:1')


    confidence_intervals.update({'Doc FirstP (DPR)': confidence_intervals.get('BM25:0 DPR:1')})
    confidence_intervals.update({'Doc (BM25)': confidence_intervals.get('BM25:1 DPR:0')})
    confidence_intervals.pop('BM25:1 DPR:0')
    confidence_intervals.pop('BM25:0 DPR:1')
    confidence_intervals.pop('BM25:1 DPR:1')

    dpr_dict_parm = read_in_run_from_pickle(dpr_file_parm)
    dpr_dict_parm = remove_query_from_ranked_list(dpr_dict_parm)

    bm25_dict_parm = read_in_run_from_pickle(bm25_file_parm)
    bm25_dict_parm = remove_query_from_ranked_list(bm25_dict_parm)

    #compare_overlap_rel(dpr_dict_parm, bm25_dict_parm, qrels)
    #analyze_score_distribution(dpr_dict_parm, 'dpr', output_dir)
    #analyze_score_distribution(bm25_dict_parm, 'bm25', output_dir)
    weights = [[1,0], [0,1], [1, 1]]

    plotting_data2 = {}
    for weight in weights:
        mode = ['test', 'separate_para_weight_dpr_{}_bm25_{}'.format(weight[0], weight[1]), 'rrf',
                'legal_task2']
        run, measures = evaluate_weight(dpr_dict_parm, bm25_dict_parm, qrels, mode, weight[0], weight[1], measurements)
        if 'recall_100' in measurements:
            plotting_data.update({'BM25:{} DPR:{}'.format(weight[1], weight[0]): create_plot_data_recall(measures)})
        else:
            plotting_data.update({'BM25:{} DPR:{}'.format(weight[1], weight[0]): create_plot_data(measures)})
        confidence_intervals.update(
                {'BM25:{} DPR:{}'.format(weight[1], weight[0]): create_ci_data(run, qrels, measurements)})

        plotting_data2.update({'BM25:{} DPR:{}'.format(weight[1], weight[0]): measures})

    print(plotting_data)
    print(plotting_data2)

    #plotting_data.update({'PARM (DPR)+PARM (BM25)': plotting_data.get('BM25:1 DPR:1')})
    plotting_data.update({'PARM-VRRF (DPR)': plotting_data.get('BM25:0 DPR:1')})
    plotting_data.update({'PARM-RRF (BM25)': plotting_data.get('BM25:1 DPR:0')})
    plotting_data.pop('BM25:1 DPR:0')
    plotting_data.pop('BM25:0 DPR:1')
    plotting_data.pop('BM25:1 DPR:1')

    confidence_intervals.update({'PARM-VRRF (DPR)': confidence_intervals.get('BM25:0 DPR:1')})
    confidence_intervals.update({'PARM-RRF (BM25)': confidence_intervals.get('BM25:1 DPR:0')})
    confidence_intervals.pop('BM25:1 DPR:0')
    confidence_intervals.pop('BM25:0 DPR:1')
    confidence_intervals.pop('BM25:1 DPR:1')

    #desired_order_list = ['PARM-VRRF (DPR)','PARM-RRF (BM25)', 'Doc (BM25)','Doc (DPR)']
    desired_order_list = ['Doc FirstP (DPR)', 'Doc (BM25)','PARM-RRF (BM25)','PARM-VRRF (DPR)']
    plotting_data = {k: plotting_data[k] for k in desired_order_list}
    confidence_intervals = {k: confidence_intervals[k] for k in desired_order_list}


    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    plt.xlabel('Cut-off', fontsize=15)
    plt.grid(True, linewidth=0.5, color='lightgrey', linestyle='-')
    plt.ylabel('Recall', fontsize=15)
    i = 0
    colours = ['#D69915','#D60038','#00D6C4','#004BD6']#['#004BD6','#00D6C4','#D60038','#D69915'] #['#0BD626','#00D6C4','#004BD6','#D60038','#D69915']
    for name, measure in plotting_data.items():
        xs, ys = zip(*measure.values())
        print(xs)
        print(ys)
        ci_min, ci_max = zip(*confidence_intervals.get(name).values())
        print(ci_min)
        print(ci_max)
        labels = measure.keys()
        # display
        plt.scatter(xs, ys, c=colours[i], marker='o', edgecolors=colours[i])
        plt.plot(xs, ys, colours[i], label=name)
        #sns.lineplot(xs, ys, ci=80, color=colours[i], label=name)
        print((list(ys) + list(ci_min)))
        ax.fill_between(xs, (ci_min), (ci_max), color=colours[i], alpha=.1)
        #sns.regplot(xs, ys, ci=80)
        # for label, x, y in zip(labels, xs, ys):
        #    plt.annotate(label, xy=(x, y))
        i += 1
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='lower right')
    #desired_order_dict = {'PARM-RRF (BM25)':2, 'PARM-VRRF (DPR)':0, 'Doc (BM25)':3, 'Doc (DPR)':4}#{'PARM (DPR)+PARM (BM25)':0, 'PARM (BM25)':2, 'PARM (DPR)':1, 'Doc (BM25)':3, 'Doc (DPR)':4}
    #by_label = dict(sorted(zip(labels, handles), key=lambda t: desired_order_dict.get(t[0])))
    #print(by_label)
    #print(by_label.keys())
    #ax.legend(by_label.values(), by_label.keys())
    ##labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: desired_order_dict.get(t[0])))
    ##ax.legend(handles, labels)
    #plt.legend(loc="lower right")
    plt.xticks(range(0, 20, 2), range(100, 1100, 100))
    ax.patch.set_facecolor('white')
    plt.savefig(os.path.join(output_dir, 'dpr_bm25_{}_dpr_parm_all17.svg'.format(mode[0])),bbox_inches='tight')




