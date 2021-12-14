import os
import pytrec_eval
import seaborn as sns
import numpy as np
import pickle
import json
sns.set(color_codes=True, font_scale=1.2)
from preprocessing.caselaw_stat_corpus import preprocess_label_file
from eval.eval_bm25 import analyze_correlations_bet_para, aggregate_run_ranks_overlap, aggregate_run_mean_score, \
    aggregate_run_overlap, aggregate_run_interleave
from eval.eval_bm25_coliee2021 import aggregate_run_rrf


def read_run_whole_doc(pred_dir: str, scores='ranks'):
    with open(pred_dir, 'r') as json_file:
        pred = json.load(json_file)

    pred_dict = {}
    for question in pred:
        question_id = question.get('answers')[0].split('_')[0]
        print(question_id)
        pred_list = {}
        i = 0
        for predition in question.get('ctxs'):
            if scores == 'scores':
                pred_list.update({predition.get('id').split('_0')[0]: float(predition.get('score'))})
            else:
                pred_list.update({predition.get('id').split('_0')[0]: float(len(question.get('ctxs')) - i)})
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
        print(question_id)
        print(question_para_id)
        pred_list = {}
        i = 0
        for prediction in question.get('ctxs'):
            if scores == 'scores':
                pred_list.update({prediction.get('id').split('_')[0]: float(prediction.get('score'))})
            else:
                pred_list.update({prediction.get('id').split('_')[0]: float(len(question.get('ctxs')) - i)})
                i += 1
        if pred_dict.get(question_id):
            pred_dict.get(question_id).update({question_para_id: pred_list})
        else:
            pred_dict.update({question_id: {}})
            pred_dict.get(question_id).update({question_para_id: pred_list})
    return pred_dict


def read_run_separate_aggregate(pred_dir: str, aggregation='interleave', scores='ranks'):
    # geh in den bm25 folder, lies in dokument und query: dann dict {query: {top 1000}}
    if aggregation == 'overlap_scores' or aggregation == 'mean_scores':
        scores = 'scores'

    run = read_run_separate(pred_dir, scores)

    # now i need an aggregation function here, different choices
    if aggregation == 'overlap_docs':
        # now aggregate according to the overlap of the docs in the paragraphs!
        print('aggregate overlapping docs')
        run_aggregated = aggregate_run_overlap(run)
    elif aggregation == 'interleave':
        print('aggregate interleave docs')
        run_aggregated = aggregate_run_interleave(run)
    elif aggregation == 'overlap_ranks':
        # now aggregate according to the overlap of the docs in the paragraphs!
        run_aggregated = aggregate_run_ranks_overlap(run)
    elif aggregation == 'overlap_scores':
        # now aggregate according to the overlap of the docs in the paragraphs!
        run_aggregated = aggregate_run_ranks_overlap(run)
    elif aggregation == 'mean_scores':
        run_aggregated = aggregate_run_mean_score(run)
    elif aggregation == 'rrf':
        run_aggregated = aggregate_run_rrf(run)
    else:
        print('no aggregation but why?')
        run_aggregated = run
    return run_aggregated


def eval_ranking_dpr(label_file, pred_dir, output_dir, output_file, aggregation='interleave', scores='ranks'):

    if aggregation == 'overlap_scores':
        scores = 'scores'

    qrels = preprocess_label_file(label_file)
    qrels_updated = {}
    for key, value in qrels.items():
        qrels_updated.update({key: {}})
        for val in value:
            qrels_updated.get(str(key)).update({str(val): 1})

    if 'separate' in pred_dir:
        print('i do separate')
        run = read_run_separate_aggregate(pred_dir, aggregation, scores)
    else:
        print('i do whole')
        run = read_run_whole_doc(pred_dir, scores)

    run_updated = {}
    for key, value in run.items():
        for key2, value2 in value.items():
            if run_updated.get(key):
                run_updated.get(key).update({'-'.join(key2.split('-')[:2]): float(value2)})
            else:
                run_updated.update({str(key): {}})
                run_updated.get(key).update({'-'.join(key2.split('-')[:2]): float(value2)})

    qrels_new = convert_ids(qrels_updated)
    run_new = convert_ids(run_updated)

    #ranking_eval(qrels, run, output_dir, output_file)
    return run_new, qrels_new


def convert_ids(run):
    run_new = {}
    for key, value in run.items():
        run_new.update({'id{}'.format(key):{}})
        for key2, value2 in value.items():
            run_new.get('id{}'.format(key)).update({'id{}'.format(key2):value2})
    return run_new


def ranking_eval(qrels, run, output_dir, measurements, output_file='eval_bm25_aggregate_overlap.txt'):
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measurements)
    # measurements)

    # {'recall_1', 'recall_2', 'recall_3', 'recall_4', 'recall_5', 'recall_6', 'recall_7', 'recall_8',
    # 'recall_9', 'recall_10','recall_11', 'recall_12', 'recall_13', 'recall_14', 'recall_15', 'recall_16', 'recall_17', 'recall_18',
    # 'recall_19', 'recall_20','P_1', 'P_2', 'P_3', 'P_4', 'P_5', 'P_6', 'P_7', 'P_8', 'P_9', 'P_10',
    # 'P_11', 'P_12', 'P_13', 'P_14', 'P_15', 'P_16', 'P_17', 'P_18', 'P_19', 'P_20'}) # {'recall_100', 'recall_200', 'recall_300', 'recall_500', 'recall_1000'})

    results = evaluator.evaluate(run)

    def print_line(measure, scope, value):
        print('{:25s}{:8s}{:.4f}'.format(measure, scope, value))

    def write_line(measure, scope, value):
        return '{:25s}{:8s}{:.4f}'.format(measure, scope, value)

    for query_id, query_measures in sorted(results.items()):
        for measure, value in sorted(query_measures.items()):
            print_line(measure, query_id, value)

        # for measure in sorted(query_measures.keys()):
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


def eval_mode(mode, measurements, label_file):
    pred_dir = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/dpr/{}/output/{}_{}_maxp_top1000.json'.format(
        mode[3], mode[0], mode[1])
    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/dpr/{}/eval'.format(mode[3])
    output_file = 'eval_dpr_{}_{}_{}_aggregation_{}.txt'.format(mode[3], mode[0], mode[1], mode[2])

    if mode[0] == 'train':
        run, qrels = eval_ranking_dpr(label_file, pred_dir, output_dir, output_file, mode[2])
        print(run, qrels)

    return run, qrels


def ranking_eval2(qrels, run, output_dir, output_file, measurements):
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measurements)
    # measurements)

    # {'recall_1', 'recall_2', 'recall_3', 'recall_4', 'recall_5', 'recall_6', 'recall_7', 'recall_8',
    # 'recall_9', 'recall_10','recall_11', 'recall_12', 'recall_13', 'recall_14', 'recall_15', 'recall_16', 'recall_17', 'recall_18',
    # 'recall_19', 'recall_20','P_1', 'P_2', 'P_3', 'P_4', 'P_5', 'P_6', 'P_7', 'P_8', 'P_9', 'P_10',
    # 'P_11', 'P_12', 'P_13', 'P_14', 'P_15', 'P_16', 'P_17', 'P_18', 'P_19', 'P_20'}) # {'recall_100', 'recall_200', 'recall_300', 'recall_500', 'recall_1000'})

    results = evaluator.evaluate(run)

    #bm25_folder = '/mnt/c/Users/salthamm/Documents/phd/data/clef-ip/2011_prior_candidate_search/bm25/search/{}/{}'.format(
    #    mode[1], mode[0])
    #output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/clef-ip/2011_prior_candidate_search/bm25/eval/{}/{}'.format(
    #    mode[1], mode[0])

    #output_file = 'test.txt'

    for key, value in results.items():
        print('{}:{}'.format(key, value))

    def print_line(measure, scope, value):
        print('{:25s}{:8s}{:.4f}'.format(measure, scope, value))

    def write_line(measure, scope, value):
        return '{:25s}{:8s}{:.4f}'.format(measure, scope, value)

    for query_id, query_measures in sorted(results.items()):
        for measure, value in sorted(query_measures.items()):
            print_line(measure, query_id, value)

        # for measure in sorted(query_measures.keys()):
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


def eval_mode2(mode, measurements, label_file):
    run, qrels = eval_mode(mode, measurements, label_file)

    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/dpr/{}/eval'.format(mode[3])
    output_file = 'eval_dpr_{}_{}_{}_aggregation_{}.txt'.format(mode[3], mode[0], mode[1], mode[2])

    ranking_eval2(qrels, run, output_dir, output_file, measurements)

    return run, qrels


def print_line(measure, scope, value):
    print('{:25s}{:8s}{:.4f}'.format(measure, scope, value))

def write_line(measure, scope, value):
    return '{:25s}{:8s}{:.4f}'.format(measure, scope, value)


if __name__ == "__main__":
    measurements = {'recall_100', 'recall_200', 'recall_300', 'recall_500', 'recall_1000', 'ndcg_cut_10', 'recip_rank'}

    label_file = '/mnt/c/Users/salthamm/Documents/coding/ussc-caselaw-collection/airs2017-collection/qrel.txt'

    #run, qrels = eval_mode2(['train', 'whole_doc', 'overlap_docs', 'legalbert'], measurements, label_file)
    #output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/dpr/legalbert/eval'
    #with open(os.path.join(output_dir, 'run_dpr_aggregate_legalbert_doc.pickle'), 'wb') as f:
    #    pickle.dump(run, f)
    #run2, qrels = eval_mode2(['train', 'whole_doc', 'rrf', 'legalbert_doc'], measurements, label_file) #maxp, firstp and rrf missing
    #run2, qrels = eval_mode2(['train', 'whole_doc', 'rrf', 'legalbert'], measurements, label_file)
    mode = ['train', 'whole_doc', 'rrf', 'legalbert_doc']
    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/dpr/legalbert_doc/eval'
    with open(os.path.join(output_dir, 'run_dpr_aggregate_firstp.pickle'), 'rb') as f:
        run = pickle.load(f)
    with open(os.path.join(output_dir, 'qrels.pickle'), 'rb') as f:
        qrels = pickle.load(f)

    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/dpr/{}/eval'.format(mode[3])
    output_file = 'eval_dpr_{}_{}_{}_aggregation_{}.txt'.format(mode[3], mode[0], mode[1], mode[2])

    ranking_eval2(qrels, run, output_dir, output_file, measurements)
    #eval_mode2(['train', 'separate_para', 'overlap_docs', 'legalbert'], measurements, label_file)
    #eval_mode2(['train', 'separate_para', 'overlap_scores', 'legalbert'], measurements, label_file)
    #eval_mode2(['train', 'separate_para', 'interleave', 'legalbert'], measurements, label_file)
    #eval_mode2(['train', 'separate_para', 'mean_scores', 'legalbert'], measurements, label_file)

    #run2, qrels2 = eval_mode2(['train', 'whole_doc', 'overlap_docs', 'bert'], measurements, label_file)
    #output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/dpr/bert/eval'
    #with open(os.path.join(output_dir, 'run_bm25_aggregate_bert_doc.pickle'), 'wb') as f:
    #    pickle.dump(run2, f)

    eval_mode2(['train', 'separate_para', 'rrf', 'bert'], measurements, label_file)
    eval_mode2(['train', 'separate_para', 'overlap_docs', 'bert'], measurements, label_file)
    eval_mode2(['train', 'separate_para', 'overlap_scores', 'bert'], measurements, label_file)
    eval_mode2(['train', 'separate_para', 'interleave', 'bert'], measurements, label_file)
    eval_mode2(['train', 'separate_para', 'mean_scores', 'bert'], measurements, label_file)