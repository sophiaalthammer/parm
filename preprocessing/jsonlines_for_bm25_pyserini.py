import os
import argparse
import jsonlines
import pickle
from preprocessing.stat_corpus import only_english, only_string_in_dict, lines_to_paragraphs


def read_in_docs(corpus_dir: str, output_dir: str, pickle_dir: str, removal=True):
    '''
    reads in all files, separates them in intro, summary and the single paragraphs, removes non-informative
    intros and summaries, writes them into jsonlines file, prints if it fails to read a certain file and
    stores it in failed_files
    :param corpus_dir: directory of the corpus containing the text files
    :param output_dir: output directory where the pickled files of the non-informative intros and summaries are
    :param removal: if non-informative text should be removed in the intros and the summaries
    :return:
    '''
    with open(os.path.join(pickle_dir, 'intro_text_often.pkl'), 'rb') as f:
        intro_often = pickle.load(f)
    with open(os.path.join(pickle_dir, 'summ_text_often.pkl'), 'rb') as f:
        summ_often = pickle.load(f)

    dict_paragraphs = {}
    failed_files = []
    for root, dirs, files in os.walk(corpus_dir):
        for file in files:
        #file = '001_001.txt'
            with open(os.path.join(corpus_dir, file), 'r') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines if line.strip('\n') is not ' ' and line.strip() is not '']
                paragraphs = lines_to_paragraphs(lines)
                if paragraphs:
                    paragraphs = only_english(paragraphs)
                    paragraphs = only_string_in_dict(paragraphs)
                    if removal:
                        if paragraphs.get('intro') in intro_often:
                            paragraphs.update({'intro': None})
                        if paragraphs.get('Summary:') in summ_often:
                            paragraphs.update({'Summary:': None})
                    dict_paragraphs.update({file.split('.')[0]: paragraphs})
                else:
                    print('reading in of file {} doesnt work'.format(file))
                    failed_files.append(file)

    #with open(os.path.join(output_dir, 'paragraphs_jsonlines.pickle'), 'wb') as f:
    #    pickle.dump(dict_paragraphs, f)
    #with open(os.path.join(output_dir, 'failed_files_jsonlines.pickle'), 'wb') as f:
    #    pickle.dump(failed_files, f)

    return dict_paragraphs, failed_files


def jsonl_index_whole_doc(output_dir: str, dict_paragraphs: dict):
    """
    creates jsonl file for bm25 index and indexes the whole document with intro and summary as one sample
    :param output_dir:
    :param dict_paragraphs:
    :return:
    """
    with jsonlines.open(os.path.join(output_dir, 'corpus_whole_docs_removed.jsonl'), mode='w') as writer:
        for key, values in dict_paragraphs.items():
            i = 0
            # values is also a dictionary with intro, summary and paragraphs
            whole_text = []
            for key2, values2 in values.items():
                if values2:
                    whole_text.append(values2)
            writer.write({'id': '{}_{}'.format(key.split('.txt')[0], i),
                          'contents': ' '.join(whole_text)})
            i += 1


def jsonl_index_doc_only_para(output_dir, dict_paragraphs):
    """
    creates jsonl file for bm25 index and indexes the whole document without intro and summary as one sample
    :param output_dir:
    :param dict_paragraphs:
    :return:
    """
    with jsonlines.open(os.path.join(output_dir, 'corpus_doc_only_para.jsonl'), mode='w') as writer:
        for key, values in dict_paragraphs.items():
            i = 0
            # values is also a dictionary with intro, summary and paragraphs
            whole_text = []
            for key2, values2 in values.items():
                if key2 != 'intro' and key2 != 'Summary:':
                    if values2:
                        whole_text.append(values2)
            writer.write({'id': '{}_{}'.format(key.split('.txt')[0], i),
                          'contents': ' '.join(whole_text)})
            i += 1


def jsonl_index_para_separately(output_dir, dict_paragraphs, intro_summ=False):
    '''
    creates jsonl file with one sample containing one passage, if intro_summ = False then it only considers
    the paragraphs, if True then it also includes the intros and summaries as samples
    :param output_dir:
    :param dict_paragraphs:
    :param intro_summ:
    :return:
    '''
    with jsonlines.open(os.path.join(output_dir, 'corpus_separately_para_{}.jsonl'.format('with_intro_summ' if intro_summ else 'only')), mode='w') as writer:
        for key, values in dict_paragraphs.items():
            i = 0
            for key2, values2 in values.items():
                if not intro_summ:
                    if key2 != 'intro' and key2 != 'Summary:':
                        if values2:
                            writer.write({'id': '{}_{}'.format(key.split('.txt')[0], i),
                                          'contents': values2})
                            i += 1
                else:
                    if values2:
                        writer.write({'id': '{}_{}'.format(key.split('.txt')[0], i),
                                      'contents': values2})
                        i += 1



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

    corpus_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2020/task1/corpus'
    pickle_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2020/task1/pickle_files'
    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2020/task1/output'
    label_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2020/task1/task1_train_2020_labels.json'
    label_file_test = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2020/task1/task1_test_2020_labels.json'
    base_case_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2020/task1/base_case_all'

    # test functions with smaller datasets
    #corpus_dir_test = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2020/task1/corpus_test'
    #output_dir_test = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2020/task1/output_test'

    dict_paragraphs, failed_files = read_in_docs(corpus_dir, output_dir, pickle_dir, removal=True)

    jsonl_index_whole_doc(output_dir, dict_paragraphs)
    jsonl_index_doc_only_para(output_dir, dict_paragraphs)
    jsonl_index_para_separately(output_dir, dict_paragraphs,
                                intro_summ=False)  # without summary and intro
    jsonl_index_para_separately(output_dir, dict_paragraphs,
                                intro_summ=True)  # with summary and intro as separate paragraphs


