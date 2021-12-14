import os
import json
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from eval.eval_bm25_coliee2021 import read_label_file #ranking_eval
from retrieval.bm25_aggregate_paragraphs import sort_write_trec_output
from preprocessing.coliee21_task2_bm25 import ranking_eval
from analysis.compare_bm25_dpr import read_in_run_from_pickle, remove_query_from_ranked_list, evaluate_weight, create_plot_data_recall, create_plot_data
from preprocessing.preprocessing_coliee_2021_task1 import lines_to_paragraphs, only_string_in_dict, only_english
#from eval.eval_dpr_coliee2021 import read_run_separate


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
                pred_list.update({prediction.get('id'): len(question.get('ctxs')) - i})
                i += 1
        if pred_dict.get(question_id):
            pred_dict.get(question_id).update({question_para_id: pred_list})
        else:
            pred_dict.update({question_id: {}})
            pred_dict.get(question_id).update({question_para_id: pred_list})
    return pred_dict


def compare_overlap_rel(dpr_dict, bm25_dict, qrels):
    # filter the dictionary for only the relevant ones
    dpr_dict_rel = return_rel_docs_for_dict(qrels, dpr_dict)
    bm25_dict_rel = return_rel_docs_for_dict(qrels, bm25_dict)

    # now compare overlap of relevant docs
    compare_overlap(dpr_dict_rel, bm25_dict_rel)


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
    bm25_rel_doc_total = []
    dpr_rel_doc_total = []
    intersect_total = []
    for query_id in dpr_dict_rel.keys():
        dpr_rel_doc = set(dpr_dict_rel.get(query_id).keys())
        bm25_rel_doc = set(bm25_dict_rel.get(query_id).keys())
        if bm25_rel_doc:
            intersection_bm25.append(len(dpr_rel_doc.intersection(bm25_rel_doc)) / len(bm25_rel_doc))
            bm25_rel_doc_total.append(len(bm25_rel_doc))
            intersect_total.append(len(dpr_rel_doc.intersection(bm25_rel_doc)))
        # else:
        #    intersection_bm25.append(0)
        if dpr_rel_doc:
            intersection_dpr.append(len(dpr_rel_doc.intersection(bm25_rel_doc)) / len(dpr_rel_doc))
            dpr_rel_doc_total.append(len(dpr_rel_doc))
        # else:
        #    intersection_dpr.append(0)

    print('average percentual intersection of bm25 results which are also found in dpr {}'.format(
        np.mean(intersection_bm25)))
    print('average percentual intersection of dpr results which are also found in bm25 {}'.format(
        np.mean(intersection_dpr)))
    print('total number of relevant found results in bm25 {}'.format(np.sum(bm25_rel_doc_total)))
    print('total number of relevant found results in dpr {}'.format(np.sum(dpr_rel_doc_total)))
    print('total number of relevant docs in intersection of bm25 and dpr {}'.format(np.sum(intersect_total)))


def first_diff_analysis(dpr_dict_parm, bm25_dict_parm, qrels):
    dpr_dict_rel = return_rel_docs_for_dict(qrels, dpr_dict_parm)
    bm25_dict_rel = return_rel_docs_for_dict(qrels, bm25_dict_parm)

    query_diff = {}
    query_diff_length = {}
    for key, value in bm25_dict_rel.items():
        query_diff.update({key: {}})
        query_diff_length.update({key: {}})
        bm25_key_dict = value
        dpr_key_dict = dpr_dict_rel.get(key)
        bm25_retrieved = set(bm25_key_dict.keys())
        dpr_retrieved = set(dpr_key_dict.keys())
        intersect = list(bm25_retrieved.intersection(dpr_retrieved))
        only_bm25 = list(bm25_retrieved.difference(dpr_retrieved))
        only_dpr = list(dpr_retrieved.difference(bm25_retrieved))
        query_diff.get(key).update({'intersect': intersect, 'only_bm25': only_bm25, 'only_dpr': only_dpr})
        query_diff_length.get(key).update(
            {'intersect': len(intersect), 'only_bm25': len(only_bm25), 'only_dpr': len(only_dpr)})

    only_bm25 = []
    only_dpr = []
    for key, value in query_diff_length.items():
        if value.get('intersect') != 0:
            share_bm25 = value.get('only_bm25') / value.get('intersect')
            share_dpr = value.get('only_dpr') / value.get('intersect')
        else:
            share_bm25 = 0
            share_dpr = 0
        #print('for document {} we have intersection of {}, only bm25 {} and only dpr {}'.format(key,
        #                                                                                        value.get('intersect'),
        #                                                                                        value.get('only_bm25'),
        #                                                                                        value.get('only_dpr')))
        #print('for document {} share of only bm25 {}, share of only dpr {}'.format(key, share_bm25, share_dpr))
        only_bm25.append(share_bm25)
        only_dpr.append(share_dpr)

    print('share of only bm25 relevant docs is {}'.format(np.mean(only_bm25)))
    print('share of only dpr relevant docs is {}'.format(np.mean(only_dpr)))

    # 0.04175213675213675
    # 0.004666666666666666
    # good these are the complement numbers to
    # 0.9697428571428571
    # 0.9966261808367071

    no_queries = 0
    # now compare overlap of relevant docs
    for key, value in query_diff_length.items():
        if value.get('only_bm25') > 0 or value.get('only_dpr') > 0:
            print('for document {} we have intersection of {}, only bm25 {} and only dpr {}'.format(key, value.get(
                'intersect'), value.get('only_bm25'), value.get('only_dpr')))
            no_queries += 1

    print('{} queries our of {} have different results for dpr and bm25'.format(no_queries, len(dpr_dict_parm.keys())))

    return dpr_dict_rel, bm25_dict_rel, query_diff, query_diff_length


def read_in_label_train_test(mode):
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
    return qrels


def read_in_file(pickle_dir, corpus_dir, id):
    with open(os.path.join(pickle_dir, 'intro_text_often.pkl'), 'rb') as f:
        intro_often = pickle.load(f)
    with open(os.path.join(pickle_dir, 'summ_text_often.pkl'), 'rb') as f:
        summ_often = pickle.load(f)

    dict_paragraphs = {}
    file = '{}.txt'.format(id)
    with open(os.path.join(corpus_dir, file), 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip('\n') is not ' ' and line.strip() is not '']
        # remove fragment supressed and \xa0
        lines = [line.replace('<FRAGMENT_SUPPRESSED>', '').strip() for line in lines if
                 line.replace('<FRAGMENT_SUPPRESSED>', '').strip() is not '']
        lines = [line.replace('\xa0', '').strip() for line in lines if
                 line.replace('\xa0', '').strip() is not '']
        # remove lines with only punctuation
        lines = [line for line in lines if re.sub(r'[^\w\s]', '', line) is not '']
        paragraphs = lines_to_paragraphs(lines)
        if paragraphs:
            paragraphs = only_english(paragraphs)
            paragraphs = only_string_in_dict(paragraphs)
            if paragraphs.get('intro') in intro_often:
                paragraphs.update({'intro': None})
            if paragraphs.get('Summary:') in summ_often:
                paragraphs.update({'Summary:': None})
            dict_paragraphs.update({file.split('.')[0]: paragraphs})
        else:
            print('reading in of file {} doesnt work'.format(file))

    return dict_paragraphs.get(id)


def get_diff_query_ids(query_diff_length):
    diff_query_ids = []
    for key, value in query_diff_length.items():
        if value.get('only_bm25') > 0 or value.get('only_dpr') > 0:
            diff_query_ids.append(key)
    return diff_query_ids


def write_case_lines(query_text, f):
    if query_text.get('intro'):
        f.write('this is the intro: {} \n \n'.format(query_text.get('intro')))
    if query_text.get('Summary:'):
        f.write('this is the summary: {} \n \n'.format(query_text.get('Summary:')))
    for key, value in query_text.items():
        if key != 'intro' and key != 'Summary:':
            f.write('this is paragraph {}: \n {} \n \n'.format(key, value))


def write_diff_cases(query_diff_length_parm, query_diff_parm, pickle_dir, output_dir, corpus_dir):
    diff_query_ids = get_diff_query_ids(query_diff_length_parm)

    for id in diff_query_ids:
        query_text = read_in_file(pickle_dir, corpus_dir, id)
        bm25_text = {}
        dpr_text = {}
        intersect_text = {}
        if query_diff_parm.get(id).get('only_bm25'):
            for id_bm25 in query_diff_parm.get(id).get('only_bm25'):
                bm25_text.update({id_bm25: read_in_file(pickle_dir, corpus_dir, id_bm25)})
        if query_diff_parm.get(id).get('only_dpr'):
            for id_dpr in query_diff_parm.get(id).get('only_dpr'):
                dpr_text.update({id_dpr: read_in_file(pickle_dir, corpus_dir, id_dpr)})
        if query_diff_parm.get(id).get('intersect'):
            for id_int in query_diff_parm.get(id).get('intersect'):
                intersect_text.update({id_int: read_in_file(pickle_dir, corpus_dir, id_int)})

        os.makedirs(os.path.join(output_dir, 'query_{}'.format(id)))
        with open(os.path.join(output_dir, 'query_{}'.format(id), 'query_{}.txt'.format(id)), 'w') as f:
            write_case_lines(query_text, f)
        for id_bm25, text in bm25_text.items():
            with open(os.path.join(output_dir, 'query_{}'.format(id), 'parm_{}.txt'.format(id_bm25)), 'w') as f:
                write_case_lines(text, f)
        for id_dpr, text in dpr_text.items():
            with open(os.path.join(output_dir, 'query_{}'.format(id), 'doc_{}.txt'.format(id_dpr)), 'w') as f:
                write_case_lines(text, f)
        for id_int, text in intersect_text.items():
            with open(os.path.join(output_dir, 'query_{}'.format(id), 'int_{}.txt'.format(id_int)), 'w') as f:
                write_case_lines(text, f)


def write_cases_for_ids(query_ids, query_diff_parm, pickle_dir, output_dir, corpus_dir):

    for id in query_ids:
        query_text = read_in_file(pickle_dir, corpus_dir, id)
        bm25_text = {}
        dpr_text = {}
        intersect_text = {}
        if query_diff_parm.get(id).get('only_bm25'):
            for id_bm25 in query_diff_parm.get(id).get('only_bm25'):
                bm25_text.update({id_bm25: read_in_file(pickle_dir, corpus_dir, id_bm25)})
        if query_diff_parm.get(id).get('only_dpr'):
            for id_dpr in query_diff_parm.get(id).get('only_dpr'):
                dpr_text.update({id_dpr: read_in_file(pickle_dir, corpus_dir, id_dpr)})
        if query_diff_parm.get(id).get('intersect'):
            for id_int in query_diff_parm.get(id).get('intersect'):
                intersect_text.update({id_int: read_in_file(pickle_dir, corpus_dir, id_int)})

        os.makedirs(os.path.join(output_dir, 'query_{}'.format(id)))
        with open(os.path.join(output_dir, 'query_{}'.format(id), 'query_{}.txt'.format(id)), 'w') as f:
            write_case_lines(query_text, f)
        for id_bm25, text in bm25_text.items():
            with open(os.path.join(output_dir, 'query_{}'.format(id), 'bm25_{}.txt'.format(id_bm25)), 'w') as f:
                write_case_lines(text, f)
        for id_dpr, text in dpr_text.items():
            with open(os.path.join(output_dir, 'query_{}'.format(id), 'dpr_{}.txt'.format(id_dpr)), 'w') as f:
                write_case_lines(text, f)
        for id_int, text in intersect_text.items():
            with open(os.path.join(output_dir, 'query_{}'.format(id), 'int_{}.txt'.format(id_int)), 'w') as f:
                write_case_lines(text, f)


def return_query_para_to_doc_para(run, query_id, doc_id):
    list_included = []
    for key, value in run.get(query_id).items():
        value_new = {}
        for key2, value2 in value.items():
            value_new.update({key2.split('_')[0]: key2.split('_')[1]})
        if doc_id in value_new.keys():
            list_included.append([key, value_new.get(doc_id)])
    return list_included


if __name__ == "__main__":
    mode = ['test', 'separate_para', 'vrrf', 'legal_task1']
    # legal task2
    #dpr_file_doc = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/dpr/{}/legalbert/aggregate/{}/run_dpr_aggregated_{}_whole_doc_overlap_docs.pickle'.format(
    #    mode[3], mode[0], mode[0])
    #dpr_file_parm = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/dpr/{}/legalbert/aggregate/{}/run_aggregated_{}_{}.pickle'.format(
    #    mode[3], mode[0], mode[0], mode[2])
    # legal task1
    dpr_file_doc = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/dpr/{}/legalbert/legal_bert_lr2e05/eval/{}/run_aggregated_{}_whole_doc_legalbert_doc.pickle'.format(
        mode[3], mode[0], mode[0])
    dpr_file_parm = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/dpr/{}/legalbert/eval/{}/run_aggregated_{}_{}_legalbert_doc.pickle'.format(
        mode[3], mode[0], mode[0], mode[2])
    bm25_file_doc = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bm25/aggregate/{}/separately_para_w_summ_intro/run_bm25_aggregated_{}_whole_doc_overlap_docs.pickle'.format(
        mode[0], mode[0])
    bm25_file_parm = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bm25/aggregate/{}/separately_para_w_summ_intro/run_aggregated_{}_rrf_overlap_ranks.pickle'.format(
        mode[0], mode[0])
    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bm25_dpr_legalbert/plot/{}'.format(mode[0])

    dpr_dict_doc = read_in_run_from_pickle(dpr_file_doc)
    dpr_dict_doc = remove_query_from_ranked_list(dpr_dict_doc)
    bm25_dict_doc = read_in_run_from_pickle(bm25_file_doc)
    bm25_dict_doc = remove_query_from_ranked_list(bm25_dict_doc)

    dpr_dict_parm = read_in_run_from_pickle(dpr_file_parm)
    dpr_dict_parm = remove_query_from_ranked_list(dpr_dict_parm)
    bm25_dict_parm = read_in_run_from_pickle(bm25_file_parm)
    bm25_dict_parm = remove_query_from_ranked_list(bm25_dict_parm)

    # read in the label files
    qrels = read_in_label_train_test(mode)

    qrels_len = []
    for key, value in qrels.items():
        qrels_len.append(len(value))
    print('number of total relevant docs is {}'.format(sum(qrels_len)))

    #print(qrels.get('034592'))

    # parm retrieval (either rrf or vrrf aggregation): dense vs lexical
    dpr_dict_parm_rel, bm25_dict_parm_rel, query_diff_parm, query_diff_length_parm = first_diff_analysis(dpr_dict_parm, bm25_dict_parm, qrels)
    compare_overlap_rel(dpr_dict_parm, bm25_dict_parm, qrels)

    # compare bm25 doc to legalbertdoc vrrf parm
    dpr_dict_parm_rel2, bm25_dict_parm_rel2, query_diff_parm2, query_diff_length_parm2 = first_diff_analysis(dpr_dict_parm, bm25_dict_doc, qrels)
    compare_overlap_rel(dpr_dict_parm, bm25_dict_doc, qrels)
    # every document gets found by legalbertvrrf parm, which gets found by bm25, but 145 additional ones

    # doc retrieval: dense vs lexical
    dpr_dict_doc_rel, bm25_dict_doc_rel, query_diff_doc, query_diff_length_doc = first_diff_analysis(dpr_dict_doc, bm25_dict_doc, qrels)
    compare_overlap_rel(dpr_dict_doc, bm25_dict_doc, qrels)

    corpus_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/corpus'
    pickle_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/pickle_files'
    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/qual_study/coliee/legalbert_doc/output/parm'
    output_dir2 = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/qual_study/coliee/legalbert_doc/output/doc'

    write_diff_cases(query_diff_length_parm, query_diff_parm, pickle_dir, output_dir, corpus_dir)
    write_diff_cases(query_diff_length_doc, query_diff_doc, pickle_dir, output_dir2, corpus_dir)

    #print(query_diff_length_parm.get('059329'))
    #print(query_diff_parm.get('059329'))
    #print(qrels.get('071207'))

    #
    # compare doc and parm for bm25
    #

    # compare doc and parm for bm25
    bm25_dict_rel_doc, bm25_dict_rel_parm, query_diff_bm25, query_diff_length_bm25 = first_diff_analysis(bm25_dict_doc,bm25_dict_parm,qrels)
    compare_overlap_rel(bm25_dict_rel_doc, bm25_dict_rel_parm, qrels)

    # compare doc and parm for dpr
    dpr_dict_rel_doc, dpr_dict_rel_parm, query_diff_dpr, query_diff_length_dpr = first_diff_analysis(dpr_dict_doc,dpr_dict_parm,qrels)
    compare_overlap_rel(dpr_dict_rel_doc, dpr_dict_rel_parm, qrels)

    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/qual_study/coliee/legalbert_doc/output/dpr'
    output_dir2 = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/qual_study/coliee/legalbert_doc/output/bm25'

    write_diff_cases(query_diff_length_dpr, query_diff_dpr, pickle_dir, output_dir, corpus_dir)
    write_diff_cases(query_diff_length_bm25, query_diff_bm25, pickle_dir, output_dir2, corpus_dir)

    #output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/qual_study/coliee/output/all_parm'
    #output_dir2 = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/qual_study/coliee/output/all_doc'

    #write_cases_for_ids(query_diff_length_parm, query_diff_parm, pickle_dir, output_dir, corpus_dir)

    print(qrels.get('071207'))


    # read in the output from legalbert search for the cases!
    pred_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/dpr/legal_task2/legalbert/output/test_separate_para_top1000.json'

    run = read_run_separate(pred_dir, 'ranks')

    print(run.get('071207'))

    # now look for document '019608', which paragaph of the query is relevant to which paragraph of the document
    query_id = '071207'
    doc_id = '019608'

    query_para_to_doc_para = return_query_para_to_doc_para(run, query_id, doc_id)

    print(query_para_to_doc_para)


