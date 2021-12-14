import os
import random
import pickle
random.seed(42)
import pytrec_eval
import numpy as np
from scipy import stats
from eval.eval_bm25_coliee2021 import read_label_file
from preprocessing.caselaw_stat_corpus import preprocess_label_file


def measure_per_query(run, qrels, measurement):
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {measurement})
    results = evaluator.evaluate(run)

    result_list = []
    for query_id, query_measures in sorted(results.items()):
        for measure, value in sorted(query_measures.items()):
            result_list.append(value)
            # print_line(measure, query_id, value)
    return result_list


def do_paired_ttest(run1, run2, qrels, measurement):
    run1_list = measure_per_query(run1, qrels, measurement)
    run2_list = measure_per_query(run2, qrels, measurement)

    stat, p = stats.ttest_rel(run1_list, run2_list)
    diff_run1_run2 = [sum(x) for x in zip(run2_list, [-x for x in run1_list])]

    effect_size = np.mean(diff_run1_run2)/np.std(diff_run1_run2)
    print('p test for {} between run1 and run2 is {}'.format(measurement, p))
    print('effect size is {}'.format(effect_size))


if __name__ == "__main__":
    # load both runs and them make a t test with them

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

    # output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/dpr/legal_task2/bert/aggregate/val'
    # with open(os.path.join(output_dir, 'run_aggregated_val_rrf.pickle'), 'rb') as f:
    #     run_rrf = pickle.load(f)
    # with open(os.path.join(output_dir, 'run_aggregated_val_min.pickle'), 'rb') as f:
    #     run_min = pickle.load(f)
    # with open(os.path.join(output_dir, 'run_aggregated_val_max.pickle'), 'rb') as f:
    #     run_max = pickle.load(f)
    # with open(os.path.join(output_dir, 'run_aggregated_val_avg.pickle'), 'rb') as f:
    #     run_avg = pickle.load(f)
    # with open(os.path.join(output_dir, 'run_aggregated_val_sum.pickle'), 'rb') as f:
    #     run_sum = pickle.load(f)
    # with open(os.path.join(output_dir, 'run_aggregated_val_vrrf.pickle'), 'rb') as f:
    #     run_vrrf = pickle.load(f)
    # with open(os.path.join(output_dir, 'run_aggregated_val_vranks.pickle'), 'rb') as f:
    #     run_vranks = pickle.load(f)
    # with open(os.path.join(output_dir, 'run_aggregated_val_vscores.pickle'), 'rb') as f:
    #     run_vscores = pickle.load(f)

    #qrels = read_label_file('/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/val/val_labels.json')
    qrels = read_label_file('/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/test/task1_test_labels_2021.json')

    measurements = ['recall_100', 'recall_500', 'recall_1000', 'ndcg_cut_10']
    # measure recall per query n (measure recall_1k)
    for measurement in measurements:
        do_paired_ttest(run_bm25_doc, run_bm25_parm, qrels, measurement)
        do_paired_ttest(run_bm25_doc, run_bert_parm_rrf, qrels, measurement)
        do_paired_ttest(run_bm25_doc, run_bert_parm_vrrf, qrels, measurement)
        do_paired_ttest(run_bm25_doc, run_legbert_para_rrf, qrels, measurement)
        do_paired_ttest(run_bm25_doc, run_legbert_para_vrrf, qrels, measurement)
        do_paired_ttest(run_bm25_doc, run_legbert_doc_rrf, qrels, measurement)
        do_paired_ttest(run_bm25_doc, run_legbert_doc_vrrf, qrels, measurement)


    # for measurement in measurements:
    #     do_paired_ttest(run_rrf, run_min, qrels, measurement)
    #     do_paired_ttest(run_rrf, run_max, qrels, measurement)
    #     do_paired_ttest(run_rrf, run_sum, qrels, measurement)
    #     do_paired_ttest(run_rrf, run_avg, qrels, measurement)
    #     do_paired_ttest(run_rrf, run_vrrf, qrels, measurement)
    #     do_paired_ttest(run_rrf, run_vscores, qrels, measurement)
    #     do_paired_ttest(run_rrf, run_vranks, qrels, measurement)

    # caselaw
    # load bm25 runs
    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/bm25/eval/'
    with open(os.path.join(output_dir, 'run_bm25_aggregate2_doc_overlap_ranks.pickle'), 'rb') as f:
       run_bm25_doc = pickle.load(f)
    with open(os.path.join(output_dir, 'run_bm25_aggregate2_rrf_overlap_ranks.pickle'), 'rb') as f:
       run_bm25_parm = pickle.load(f)

    # load bert para based runs
    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/dpr/bert/eval/'
    with open(os.path.join(output_dir, 'run_dpr_aggregate_bert_rrf_overlap_ranks.pickle'), 'rb') as f:
       run_bert_parm_rrf = pickle.load(f)
    with open(os.path.join(output_dir, 'run_aggregated_train_vrrf.pickle'), 'rb') as f:
       run_bert_parm_vrrf = pickle.load(f)

    # load legalbert para based runs
    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/dpr/legalbert/eval/'
    with open(os.path.join(output_dir, 'run_dpr_aggregate_legalbert_rrf_overlap_ranks.pickle'), 'rb') as f:
       run_legbert_para_rrf = pickle.load(f)
    with open(os.path.join(output_dir, 'run_aggregated_train_vrrf.pickle'), 'rb') as f:
       run_legbert_para_vrrf = pickle.load(f)

    # load legalbert doc based runs
    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/dpr/legalbert_doc/eval/'
    with open(os.path.join(output_dir, 'run_dpr_aggregate_rrf.pickle'), 'rb') as f:
       run_legbert_doc_rrf = pickle.load(f)
    with open(os.path.join(output_dir, 'run_aggregated_train_vrrf.pickle'), 'rb') as f:
       run_legbert_doc_vrrf = pickle.load(f)

    label_file = '/mnt/c/Users/salthamm/Documents/coding/ussc-caselaw-collection/airs2017-collection/qrel.txt'
    qrels = preprocess_label_file(label_file)
    qrels_updated = {}
    for key, value in qrels.items():
        qrels_updated.update({key: {}})
        for val in value:
            qrels_updated.get(str(key)).update({str(val): 1})
    qrels = qrels_updated

    measurements = ['ndcg_cut_10', 'recall_100', 'recall_500', 'recall_1000']
    # measure recall per query n (measure recall_1k)
    for measurement in measurements:
        do_paired_ttest(run_bm25_doc, run_bm25_parm, qrels, measurement)
        do_paired_ttest(run_bm25_doc, run_bert_parm_rrf, qrels, measurement)
        do_paired_ttest(run_bm25_doc, run_bert_parm_vrrf, qrels, measurement)
        do_paired_ttest(run_bm25_doc, run_legbert_para_rrf, qrels, measurement)
        do_paired_ttest(run_bm25_doc, run_legbert_para_vrrf, qrels, measurement)
        do_paired_ttest(run_bm25_doc, run_legbert_doc_rrf, qrels, measurement)
        do_paired_ttest(run_bm25_doc, run_legbert_doc_vrrf, qrels, measurement)



    # use measure per query for your analysis!



