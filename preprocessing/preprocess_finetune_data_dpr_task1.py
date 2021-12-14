import os
import argparse
import json
import random
from math import floor
from rank_bm25 import BM25Okapi
from eval.eval_bm25_coliee2021 import read_label_file
from preprocessing.preprocessing_coliee_2021_task1 import read_in_docs
from preprocessing.preprocess_finetune_data_dpr import write_to_json
random.seed(43)


def read_in_samples_task1(dict_paragraphs, qrels, bm25_dir, no_hard_neg_docs):
    samples = []
    for query_id in qrels.keys():
        print('now we start with this query {}'.format(query_id))
        paragraph_id = 0
        for paragraph in dict_paragraphs.get(query_id):
            if dict_paragraphs.get(query_id).get(paragraph):
                try:
                    query_text = dict_paragraphs.get(query_id).get(paragraph)

                    print('written in the query text')

                    positive_ctxs = []
                    for rel_id in qrels.get(query_id).keys():
                        doc_rel_dict = dict_paragraphs.get(rel_id)
                        i = 0
                        for x, doc_rel_text in doc_rel_dict.items():
                            if doc_rel_text:
                                ctx = {"title": "",
                                       "text": doc_rel_text,
                                       # "score": 0,
                                       "psg_id": '{}_{}'.format(rel_id, i)}
                                positive_ctxs.append(ctx)
                                i += 1
                    print('done with positives')

                    # read in bm25 scores for hard negatives
                    with open(os.path.join(bm25_dir, 'bm25_top1000_{}_{}_separately_para_w_summ_intro.txt'.format(query_id,
                                                                                                                  paragraph_id)),
                              "r",
                              encoding="utf8") as out_file:
                        top1000 = out_file.read().splitlines()[:no_hard_neg_docs]
                    top1000_dict = {}
                    for top in top1000:
                        top1000_dict.update({top.split(' ')[0]: float(top.split(' ')[-1].strip())})
                    print(top1000_dict)

                    print('read in the negatives')
                    print(qrels.get(query_id))

                    hard_negative_ctxs = []
                    # irrel_ids are the first no_hard_neg_docs which are retrieved by bm25 but are not relevant
                    irrel_ids = []
                    for key, score in top1000_dict.items():
                        if key.split('_')[0] not in qrels.get(query_id).keys():
                            irrel_ids.append(key)

                    print('these are my irrelevant ones {}'.format(irrel_ids))

                    for irrel_id in irrel_ids:
                        j = 0
                        for x, doc_irrel_text in dict_paragraphs.get(irrel_id.split('_')[0]).items():
                            if doc_irrel_text:
                                ctx = {"title": "",
                                       "text": doc_irrel_text,
                                       "score": top1000_dict.get(irrel_id),
                                       "psg_id": '{}'.format(irrel_id)}
                                hard_negative_ctxs.append(ctx)

                    print('now im finished with the irrelids ')

                    # sort list of hard-negatives by score? yes do it!
                    hard_negative_ctxs.sort(key=lambda hard_negative_ctxs: hard_negative_ctxs.get('score'),
                                            reverse=True)

                    print('now i sorted the hard negatives')

                    sample = {"question": query_text,
                              "answers": ['{}_{}'.format(query_id, paragraph_id)],
                              "positive_ctxs": positive_ctxs,
                              "negative_ctxs": [],
                              "hard_negative_ctxs": hard_negative_ctxs
                              }

                    samples.append(sample)
                except:
                    print('it didnt work for this file {}_{}'.format(query_id, paragraph_id))
                print(paragraph_id)
                paragraph_id += 1
            else:
                print('it didnt work for this paragraph {} {}'.format(query_id, paragraph_id))
        print('finished with query {}'.format(query_id))
    return samples


def read_in_samples_task1_random_neg(dict_paragraphs, qrels):
    samples = []
    for query_id in qrels.keys():
        print('now we start with this query {}'.format(query_id))
        paragraph_id = 0
        for paragraph in dict_paragraphs.get(query_id):
            if dict_paragraphs.get(query_id).get(paragraph):
                try:
                    query_text = dict_paragraphs.get(query_id).get(paragraph)

                    positive_ctxs = []
                    for rel_id in qrels.get(query_id).keys():
                        doc_rel_dict = dict_paragraphs.get(rel_id)
                        i = 0
                        for x, doc_rel_text in doc_rel_dict.items():
                            if doc_rel_text:
                                ctx = {"title": "",
                                       "text": doc_rel_text,
                                       # "score": 0,
                                       "psg_id": '{}_{}'.format(rel_id, i)}
                                positive_ctxs.append(ctx)
                                i += 1

                    irrel_keys = random.sample(list(dict_paragraphs.keys()), len(qrels.get(query_id).keys()))
                    hard_negative_ctxs = []
                    # irrel_ids are the first no_hard_neg_docs which are retrieved by bm25 but are not relevant
                    irrel_ids = []
                    for key in irrel_keys:
                        if key.split('_')[0] not in qrels.get(query_id).keys():
                            irrel_ids.append(key)

                    if irrel_ids == []:
                        irrel_keys = random.sample(list(dict_paragraphs.keys()), len(qrels.get(query_id).keys())*2)
                        hard_negative_ctxs = []
                        # irrel_ids are the first no_hard_neg_docs which are retrieved by bm25 but are not relevant
                        irrel_ids = []
                        for key in irrel_keys:
                            if key.split('_')[0] not in qrels.get(query_id).keys():
                                irrel_ids.append(key)

                    print('these are my irrelevant ones {}'.format(irrel_ids))

                    for irrel_id in irrel_ids:
                        j = 0
                        for x, doc_irrel_text in dict_paragraphs.get(irrel_id.split('_')[0]).items():
                            if doc_irrel_text:
                                ctx = {"title": "",
                                       "text": doc_irrel_text,
                                       "psg_id": '{}'.format(irrel_id)}
                                hard_negative_ctxs.append(ctx)

                    sample = {"question": query_text,
                              "answers": ['{}_{}'.format(query_id, paragraph_id)],
                              "positive_ctxs": positive_ctxs,
                              "negative_ctxs": [],
                              "hard_negative_ctxs": hard_negative_ctxs
                              }

                    samples.append(sample)
                except:
                    print('it didnt work for this file {}_{}'.format(query_id, paragraph_id))
                print(paragraph_id)
                paragraph_id += 1
            else:
                print('it didnt work for this paragraph {} {}'.format(query_id, paragraph_id))
        print('finished with query {}'.format(query_id))
    return samples


def read_in_samples_task1_randneg_bm25pos(dict_paragraphs, qrels, cut_value):
    samples = []
    for query_id in qrels.keys():
        print('now we start with this query {}'.format(query_id))
        paragraph_id = 0
        for paragraph in dict_paragraphs.get(query_id):
            if dict_paragraphs.get(query_id).get(paragraph):
                try:
                    query_text = dict_paragraphs.get(query_id).get(paragraph)

                    positive_ctxs = []
                    corpus_rel_id = []
                    for rel_id in qrels.get(query_id).keys():
                        doc_rel_dict = dict_paragraphs.get(rel_id)
                        for x, doc_rel_text in doc_rel_dict.items():
                            if doc_rel_text:
                                corpus_rel_id.append(doc_rel_text)

                    tokenized_corpus = [doc.split(" ") for doc in corpus_rel_id]
                    bm25 = BM25Okapi(tokenized_corpus)
                    tokenized_query = query_text.split(" ")

                    doc_scores = bm25.get_scores(tokenized_query)

                    for i in range(0, len(corpus_rel_id)):
                        ctx = {"title": "",
                               "text": corpus_rel_id[i],
                               "score": doc_scores[i]}
                        positive_ctxs.append(ctx)

                    positive_ctxs_sorted = sorted(positive_ctxs, key=lambda k: k['score'], reverse=True)
                    positive_ctxs_cut = positive_ctxs_sorted[:cut_value]

                    irrel_keys = random.sample(list(dict_paragraphs.keys()), len(qrels.get(query_id).keys()))
                    hard_negative_ctxs = []
                    # irrel_ids are the first no_hard_neg_docs which are retrieved by bm25 but are not relevant
                    irrel_ids = []
                    for key in irrel_keys:
                        if key.split('_')[0] not in qrels.get(query_id).keys():
                            irrel_ids.append(key)

                    if irrel_ids == []:
                        irrel_keys = random.sample(list(dict_paragraphs.keys()), len(qrels.get(query_id).keys()) * 2)
                        hard_negative_ctxs = []
                        # irrel_ids are the first no_hard_neg_docs which are retrieved by bm25 but are not relevant
                        irrel_ids = []
                        for key in irrel_keys:
                            if key.split('_')[0] not in qrels.get(query_id).keys():
                                irrel_ids.append(key)

                    print('these are my irrelevant ones {}'.format(irrel_ids))

                    for irrel_id in irrel_ids:
                        for x, doc_irrel_text in dict_paragraphs.get(irrel_id.split('_')[0]).items():
                            if doc_irrel_text:
                                ctx = {"title": "",
                                       "text": doc_irrel_text,
                                       "psg_id": '{}'.format(irrel_id)}
                                hard_negative_ctxs.append(ctx)

                    sample = {"question": query_text,
                              "answers": ['{}_{}'.format(query_id, paragraph_id)],
                              "positive_ctxs": positive_ctxs_cut,
                              "negative_ctxs": [],
                              "hard_negative_ctxs": hard_negative_ctxs
                              }

                    samples.append(sample)
                except:
                    print('it didnt work for this file {}_{}'.format(query_id, paragraph_id))
                print(paragraph_id)
                paragraph_id += 1
            else:
                print('it didnt work for this paragraph {} {}'.format(query_id, paragraph_id))
        print('finished with query {}'.format(query_id))
    return samples


def resort_batch(samples, batch_size):
    sorted_index = []
    for i in range(0, len(samples)):
        sorted_index.append(floor(i / batch_size) + (i % batch_size) * batch_size)

    return [samples[i] for i in sorted_index]


if __name__ == "__main__":
    mode = ['val']
    corpus_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/corpus'
    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/finetune_dpr/{}'.format(mode[0])
    pickle_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/pickle_files'
    bm25_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bm25/search/{}/separately_para_w_summ_intro'.format(mode[0])

    label_file_train = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/train/train_wo_val_labels.json'
    label_file_val = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/val/val_labels.json'
    label_file_test = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/test/test_no_labels.json'

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

    no_hard_neg_docs = 5
    cut_value = 20
    #samples = read_in_samples_task1(dict_paragraphs, qrels, bm25_dir, no_hard_neg_docs)
    samples = read_in_samples_task1_random_neg(dict_paragraphs, qrels)
    #samples = read_in_samples_task1_randneg_bm25pos(dict_paragraphs, qrels, cut_value)

    if mode[0] == 'train':
        # print first 45 sample query ids:
        query_ids = []
        for i in range(0, 50):
            query_ids.append(samples[i].get('answers')[0])
        print(query_ids)

        # stop i need reshuffle training samples, so that all 44 times are shifted!
        # not for the val samples
        samples = resort_batch(samples, 44)

        query_ids = []
        for i in range(0, 50):
            query_ids.append(samples[i].get('answers')[0])
        print(query_ids)

    write_to_json(samples, output_dir)




