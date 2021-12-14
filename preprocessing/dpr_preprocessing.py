import os
import argparse
import jsonlines
import json
import pickle
import csv
from preprocessing.jsonlines_for_bm25_pyserini import jsonl_index_para_separately, jsonl_index_doc_only_para, jsonl_index_whole_doc #read_in_docs
#from preprocessing.stat_corpus import preprocess_label_file
from preprocessing.preprocessing_coliee_2021_task1 import read_in_docs, preprocess_label_file


def corpus_to_ctx_file(jsonl_file: str, out_file: str):
    # from corpus jsonlines file to ctx_file format: .tsv file, id, text, title
    with open(out_file, 'wt') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t')
        tsv_writer.writerow(['id', 'text', 'title'])
        with jsonlines.open(jsonl_file) as reader:
            for obj in reader:
                #print(obj.get('contents').replace('"', ""))
                tsv_writer.writerow([obj.get('id'), obj.get('contents').replace('"', ""), ''])


def base_case_to_qa_file(dict_paragraphs: dict, out_file: str, separate=True):
    # from corpus jsonlines file to ctx_file format: .tsv file, id, text, title
    with open(out_file, 'wt') as tsv_file:
        #writer = csv.writer(tsv_file, delimiter='\t', dialect="excel", lineterminator="\n")
        writer = csv.writer(tsv_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for key, values in dict_paragraphs.items():
            i = 0
            # values is also a dictionary with intro, summary and paragraphs
            if not separate:
                whole_text = []
                for key2, values2 in values.items():
                    if values2:
                        whole_text.append(values2.replace(';', ','))
                writer.writerow([' '.join(whole_text), ['{}_{}'.format(key.split('.txt')[0], i)]])
                i += 1
            else:
                for key2, values2 in values.items():
                    if values2:
                        if key.split('.txt')[0] == '001' and i == 0:
                            print(values2)
                        writer.writerow([str(values2.replace(';', ',')), ['{}_{}'.format(key.split('.txt')[0], i)]])
                        i += 1


def qa_for_label_file(dict_paragraphs: dict, label_file2: str, output_dir: str, separate=True):

    with open(label_file2, 'r') as fp:
        labels = json.load(fp)

    #labels = preprocess_label_file(label_file2)

    dict_paragraphs_train = {}
    for key in labels.keys():
        dict_paragraphs_train.update({key: dict_paragraphs.get(key)})

    base_case_to_qa_file(dict_paragraphs_train, output_dir, separate=separate)


if __name__ == "__main__":
    #
    # config
    #
    #parser = argparse.ArgumentParser()

    #parser.add_argument('--corpus-dir', action='store', dest='corpus_dir',
    #                    help='corpus directory location', required=True)
    # parser.add_argument('--output-dir', action='store', dest='output_dir',
    #                    help='output directory location', required=True)
    # parser.add_argument('--label-file-train', action='store', dest='label_file',
    #                    help='label file train', required=True)
    # parser.add_argument('--label-file-test', action='store', dest='label_file_test',
    #                    help='label file test', required=True)

    #args = parser.parse_args()

    #jsonl_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2020/task1/corpus_jsonl/corpus_separately_para_w_intro_summ.jsonl'
    #out_file = '/mnt/c/Users/salthamm/Documents/coding/DPR/data/coliee2020_corpus_separate_para/ctx_corpus_separate_para.tsv'

    #jsonl_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/corpus_jsonl/whole_doc_w_summ_intro.jsonl'
    #out_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/ctx_files/ctx_whole_doc_w_intro_summ.tsv'

    #corpus_to_ctx_file(jsonl_file, out_file)

    # now create qa file with topics from train/dev/test

    pickle_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/pickle_files'
    label_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/train/train_wo_val_labels.json'
    label_file_val = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/val/val_labels.json'
    label_file_test = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/test/test_no_labels.json'
    corpus_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/corpus'
    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/ctx_files/test/qa_whole_doc.csv'
    output_dir2 = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/ctx_files/test/qa_separate_para.csv'

    dict_paragraphs, failed_files = read_in_docs(corpus_dir, output_dir, pickle_dir, removal=True)

    #labels = preprocess_label_file(pickle_dir, label_file)
    #labels_only_train = {}
    #for i in ["{0:03}".format(i) for i in range(1,421)]:
    #    labels_only_train.update({i: labels.get(i)})

    #with open(label_file2, 'w') as fp:
    #    json.dump(labels_only_train, fp)

    qa_for_label_file(dict_paragraphs, label_file_test, output_dir, separate=False)
    qa_for_label_file(dict_paragraphs, label_file_test, output_dir2, separate=True)



    # entweder whole doc oder separate writing



