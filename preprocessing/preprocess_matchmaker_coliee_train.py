import os
import json
import random
import csv
from eval.eval_bm25_coliee2021 import read_label_file
from preprocessing.preprocessing_coliee_2021_task1 import read_in_docs
from preprocessing.coliee21_bert_pli_json import read_run
random.seed(43)


def read_in_samples_task1_random_neg_whole_doc(dict_paragraphs: dict, qrels: dict):
    # join dict_paragraphs to docs
    dict_whole_docs = dict_para_to_whole_docs(dict_paragraphs)

    samples = []
    for query_id in qrels.keys():
        print('now we start with this query {}'.format(query_id))
        query_text = dict_whole_docs.get(query_id)
        query_text = str(query_text.replace(';', ',').replace("\'s ", 's ').replace('/', ''))
        for rel_id in qrels.get(query_id).keys():
            #try:
            doc_rel_text = dict_whole_docs.get(rel_id)
            doc_rel_text = str(doc_rel_text.replace(';', ',').replace('\'s ', 's ').replace('/', ''))
            irrel_keys = random.sample(list(dict_whole_docs.keys()), 10)
            # irrel_ids are the first no_hard_neg_docs which are retrieved by bm25 but are not relevant
            irrel_ids = []
            for key in irrel_keys:
                if key not in qrels.get(query_id).keys():
                    irrel_ids.append(key)

            irrel_id = random.sample(irrel_keys, 1)
            irrel_id = irrel_id[0]
            doc_irrel_text = dict_whole_docs.get(irrel_id)
            doc_irrel_text = str(doc_irrel_text.replace(";",',').replace('\'s ', 's ').replace('/', ''))

            sample = {"question_text": query_text,
                          "pos_text": doc_rel_text,
                          "neg_text": doc_irrel_text
                          }
            print(sample)
            samples.append(sample)
            #except:
            #    print('error with the rel-id {} for query {}'.format(rel_id, query_id))
        print('finished with query {}'.format(query_id))
    return samples


def read_in_samples_task1_test_whole_doc(dict_paragraphs: dict, qrels: dict):
    # join dict_paragraphs to docs
    dict_whole_docs = dict_para_to_whole_docs(dict_paragraphs)

    samples = []
    for query_id in qrels:
        print('now we start with this query {}'.format(query_id))
        query_text = dict_whole_docs.get(query_id)
        query_text = str(query_text.replace(';', ',').replace("\'s ", 's ').replace('/', ''))
        sample = {"query_id": query_id,
                  "query_text": query_text}
        samples.append(sample)
        print('finished with query {}'.format(query_id))
    return samples


def read_samples_from_run(dict_paragraphs: dict, run: dict):
    dict_whole_docs = dict_para_to_whole_docs(dict_paragraphs)

    samples = []
    for q_id, candidate in run.items():
        query_text = dict_whole_docs.get(q_id)
        query_text = str(query_text.replace(';', ',').replace("\'s ", 's ').replace('/', ''))
        for c_id, score in candidate.items():
            candidate_text = dict_whole_docs.get(c_id)
            candidate_text = str(candidate_text.replace(';', ',').replace("\'s ", 's ').replace('/', ''))
            if candidate_text:
                samples.append({'q_id': q_id,
                                'c_id': c_id,
                                'query_text': query_text,
                                'candidate_text': candidate_text})
            else:
                print('couldnt find {} in corpus'.format(c_id))
        print('finished with query {}'.format(q_id))
    return samples


def dict_para_to_whole_docs(dict_paragraphs: dict):
    dict_whole_docs = {}
    for key, values in dict_paragraphs.items():
        i = 0
        # values is also a dictionary with intro, summary and paragraphs
        whole_text = []
        for key2, values2 in values.items():
            if values2:
                whole_text.append(values2)
        dict_whole_docs.update({key: ' '.join(whole_text)})
    return dict_whole_docs


def write_to_tsv(samples: list, out_file: str):
    with open(out_file, 'wt') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t')
        keys = list(samples[0].keys())
        for sample in samples:
            tsv_writer.writerow([sample.get(key) for key in keys])


if __name__ == "__main__":
    mode = ['test']
    corpus_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/corpus'
    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/matchmaker/rerank/data/{}'.format(mode[0])
    pickle_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/pickle_files'
    bm25_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bm25/search/{}/whole_doc_w_summ_intro'.format(mode[0])

    label_file_train = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/train/train_wo_val_labels.json'
    label_file_val = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/val/val_labels.json'
    #label_file_test = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/test/test_no_labels.json'
    label_file_test = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/test/task1_test_labels_2021.json'

    # first read in all files in corpus, non-informative parts are removed
    dict_paragraphs, failed_files = read_in_docs(corpus_dir, output_dir, pickle_dir, removal=True)

    print('finished with reading in data')

    # then read in the labels, because these are the questions and the answers, the hard negatives are sampled randomly?
    # or from the bm25 candidates? i think i should take one sample from bm25

    if mode[0] == 'train':
        qrels = read_label_file(label_file_train)
    elif mode[0] == 'val':
        qrels = read_label_file(label_file_val)
    elif mode[0] == 'test':
        with open(label_file_test, 'rb') as f:
            qrels = json.load(f)
            qrels = [x.split('.txt')[0] for x in qrels]

    #if mode[0] == 'test':
    #    samples = read_in_samples_task1_test_whole_doc(dict_paragraphs, qrels)
    #else:
    #    samples = read_in_samples_task1_random_neg_whole_doc(dict_paragraphs, qrels)

    #if mode[0] == 'train':
    #    random.shuffle(samples)

    # seba tsv file for queries
    out_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/matchmaker/rerank/data/{}/eval_{}_top1000.tsv'.format(mode[0], mode[0])

    #write_to_tsv(samples, out_file)

    # validation and test augment with top1000 from a run!
    run_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bert-pli-rerank/runs/{}/legalbert_para_vrrf_{}.pickle'.format(mode[0], mode[0])
    run = read_run(run_file, 1000)

    samples = read_samples_from_run(dict_paragraphs, run)

    write_to_tsv(samples, out_file)

    print(samples[0])





