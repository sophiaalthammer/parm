import os
import argparse
import jsonlines
import json
import pickle
import matplotlib.pyplot as plt
import csv
from preprocessing.dpr_preprocessing import corpus_to_ctx_file
from eval.eval_bm25_coliee2021 import read_label_file
from preprocessing.coliee21_task2_bm25 import ranking_eval, eval_ranking_bm25
from analysis.compare_bm25_dpr import evaluate_weight, sort_write_trec_output, read_in_aggregated_scores, \
    compare_overlap_rel, analyze_score_distribution


def entailed_fragment_to_qa(train_dir):
    list_dir = [x for x in os.walk(train_dir)]
    for sub_dir in list_dir[0][1]:
        with open(os.path.join(train_dir, sub_dir, 'qa.csv'), 'wt') as tsv_file:
            writer = csv.writer(tsv_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            with open(os.path.join(train_dir, sub_dir, 'entailed_fragment.txt'), 'r') as entailed_fragment:
                query_text_lines = entailed_fragment.read().splitlines()
                query_text = ' '.join([text.strip().replace('\n', '') for text in query_text_lines])
                writer.writerow([query_text, [str(sub_dir)]])


def read_run_whole_doc(pred_dir: str, scores='ranks'):
    # geh in den bm25 folder, lies in dokument und query: dann dict {query: {top 1000}}
    run = {}
    for root, dirs, files in os.walk(pred_dir):
        for file in files:
            with open(os.path.join(pred_dir, file), 'r') as json_file:
                pred = json.load(json_file)

                for question in pred:
                    question_id = question.get('answers')[0].split('_')[0]
                    pred_list = {}
                    i = 0
                    for predition in question.get('ctxs'):
                        if scores == 'scores':
                            pred_list.update({predition.get('id').split('_')[1]: float(predition.get('score'))})
                        else:
                            pred_list.update({predition.get('id').split('_')[1]: len(question.get('ctxs')) - i})
                            i += 1
                    run.update({question_id: pred_list})
        return run


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


if __name__ == "__main__":
    mode = ['val', False]
    train_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task2/{}'.format(mode[0])

    # create corpus for each sub_dir
    #list_dir = [x for x in os.walk(train_dir)]
    #for sub_dir in list_dir[0][1]:
    #    jsonl_file = os.path.join(train_dir, sub_dir, 'candidates.jsonl')
    #    out_file = os.path.join(train_dir, sub_dir, 'ctx_candidates.tsv')
    #    corpus_to_ctx_file(jsonl_file, out_file)

    # create qa files for each sub_dir
    #entailed_fragment_to_qa(train_dir)

    # evaluate
    if mode[0] == 'train':
        label_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task2/task2_train_wo_val_labels_2021.json'
    elif mode[0] == 'val':
        label_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task2/task2_val_labels_2021.json'

    pred_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task2/dpr/output/train_wo_val/{}'.format(mode[0])
    eval_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task2/dpr/eval/train_wo_val/{}'.format(mode[0])
    output_file = 'eval_dpr_{}'.format(mode[0])

    # evaluate and plot the recall and precision
    qrels = read_label_file(label_file)
    run = read_run_whole_doc(pred_dir, 'scores')

    measures = ranking_eval(qrels, run, eval_dir, output_file)
    #plot_recall_precision(measures, eval_dir)


    # bm25 measures!
    train_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task2/{}'.format(mode[0])
    top_n = 50
    bm25_folder = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task2/bm25/search/{}/whole_doc_{}'.format(
        mode[0], mode[1])

    # evaluate bm25
    eval_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task2/bm25/eval/{}/whole_doc_{}'.format(mode[0], mode[1])
    output_file = 'eval_bm25_recall_{}_whole_doc_{}'.format(mode[0], mode[1])

    if mode[0] == 'train':
        label_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task2/task2_train_wo_val_labels_2021.json'
    elif mode[0] == 'val':
        label_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task2/task2_val_labels_2021.json'

    # evaluate ranking of bm25 as if it was for whole documents -> no aggregation needed
    measures_bm25 = eval_ranking_bm25(label_file, bm25_folder, eval_dir, output_file)

    plotting_data_dpr = create_plot_data(measures)
    plotting_data_bm25 = create_plot_data(measures_bm25)

    plotting_data = {'DPR':plotting_data_dpr, 'BM25':plotting_data_bm25}
    #plot_measures(plotting_data, eval_dir, 'prec_rec_bm25_dpr_comparison.svg')

    # also add the weigthed files to plot!
    mode = ['val', 'weighting_1_4', 'scores']
    dpr_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task2/dpr/aggregate/{}/search_{}_something_aggregation_scores.txt'.format(
        mode[0], mode[0])
    bm25_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task2/bm25/aggregate/{}/search_{}_something_aggregation_overlap_scores.txt'.format(
        mode[0], mode[0])
    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task2/plots'

    dpr_dict = read_in_aggregated_scores(dpr_file, 50)
    bm25_dict = read_in_aggregated_scores(bm25_file, 50)

    bm25_dict_new = {}
    for key, value in bm25_dict.items():
        bm25_dict_new.update({key:{}})
        for key2, value2 in value.items():
            bm25_dict_new.get(key).update({key2.split('_')[1]:value2})

    # check if both dictionaries contain the same query ids
    dpr_keys = list(dpr_dict.keys())
    dpr_keys.sort()
    assert dpr_keys == list(bm25_dict_new.keys())

    # read in the label files
    label_file_val = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task2/task2_val_labels_2021.json'

    if mode[0] == 'val':
        qrels = read_label_file(label_file_val)

    #compare_overlap_rel(dpr_dict, bm25_dict_new, qrels)

    # analyze score distribution
    #analyze_score_distribution(dpr_dict, 'dpr', output_dir)
    #analyze_score_distribution(bm25_dict_new, 'bm25', output_dir)
    weights = [[1, 1], [1, 2], [1, 3], [1, 4]]
    for weight in weights:
        run, measures = evaluate_weight(dpr_dict, bm25_dict_new, qrels, mode, weight[0], weight[1])
        # write out in trec format the aggregated list with the combined weights!
        # best is: overlap_ranks, weight_dpr=1 weight_bm25=3
        output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task2/bm25_dpr/eval/{}/'.format(mode[0])
        sort_write_trec_output(run, output_dir, mode)

    # same plotting graph for dpr and bm25 for task1, same graph of f1 score for task1!
    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task2/bm25_dpr/eval/{}/'.format(mode[0])

    plotting_data_weighted = {}
    for weight in weights:
        with open(os.path.join(output_dir, 'measures_bm25_dpr_weigting_{}_{}.json'.format(weight[0], weight[1])),
                  'r') as f:
            measures_weight = json.load(f)
        plotting_data_weighted.update({'BM25:{} DPR:{}'.format(weight[1], weight[0]):create_plot_data(measures_weight)})

    plotting_data_weighted.update({'DPR':plotting_data_dpr, 'BM25':plotting_data_bm25})
    plot_measures(plotting_data_weighted, output_dir, 'prec_rec_bm25_dpr_weighting3.svg')
    calculcate_f1_score(plotting_data_weighted, output_dir, 'f1_scores_bm25_dpr.txt')

    plot_f1_score(plotting_data_weighted, output_dir, 'f1_scores_bm25_dpr_weighting.svg')







