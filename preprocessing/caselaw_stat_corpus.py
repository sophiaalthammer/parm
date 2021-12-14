import os
import shutil
import pickle
import jsonlines
import json
import csv
import numpy as np
from operator import itemgetter
from itertools import groupby
from preprocessing.stat_corpus import count_doc, count_words, plot_hist, analyze_text_passages
from preprocessing.dpr_preprocessing import corpus_to_ctx_file


def analyze_corpus_in_numbers(lengths, dict_paragraphs, labels_train, output_dir):
    """
    Analyzes the corpus with respect to numbers and lengths of introduction, summaries and paragraphs as well as labels
    :param lengths:
    :param dict_paragraphs:
    :param labels_train:
    :param labels_test:
    :return:
    """
    print('number of files in corpus {}'.format(len(lengths.keys())))

    avg_length = []
    for key, value in lengths.items():
        if value.get('intro'):
            intro_len = value.get('intro')
        else:
            intro_len = 0
        if value.get('lengths_paragraphs'):
            para_len = sum([x for x in value.get('lengths_paragraphs') if x])
        else:
            para_len = 0
        avg_length.append(intro_len + para_len)
    print('the documents have an average length of {}'.format(np.mean(avg_length)))
    print('the documents have an min length of {}'.format(np.min(avg_length)))
    print('the documents have an max length of {}'.format(np.max(avg_length)))
    # paragraphs
    print('average number of paragraphs per document {}'.format(
        np.mean([len(value.get('lengths_paragraphs')) for key, value in lengths.items()])))
    para = []
    for key, value in lengths.items():
        if value.get('lengths_paragraphs'):
            list = value.get('lengths_paragraphs')
            list_wo_none = [x for x in list if x]
        para.extend(list_wo_none)
    print('the paragraphs have an average length of {}'.format(np.mean(para)))
    print('the shortest paragraph has {} words'.format(np.min(para)))
    print('the longest paragraph has {} words'.format(np.max(para)))
    plot_hist(np.array([x for x in para if x<1000]), 'number of words', 'Paragraph length distribution', output_dir)
    print('there are in total {} paragraphs'.format(len(para)))

    # labels
    print('average number of relevant documents for train {}'.format(
        np.mean([len(value) for value in labels_train.values()])))


def preprocess_label_file(label_file):
    # reads in the qrel and converts it in right format
    qrels = {}
    with open(label_file, 'r') as f:
        lines = f.read().splitlines()

        for line in lines:
            splitted_line = line.split(' ')
            if qrels.get(splitted_line[0]):
                if splitted_line[3] == '1':
                    rel_cases = qrels.get(splitted_line[0])
                    rel_cases.append(splitted_line[2])
                    qrels.update({splitted_line[0]: rel_cases})
            else:
                qrels.update({splitted_line[0]: []})
                if splitted_line[3] == '1':
                    rel_cases = qrels.get(splitted_line[0])
                    rel_cases.append(splitted_line[2])
                    qrels.update({splitted_line[0]: rel_cases})

    return qrels


def read_in_para_lengths(corpus_dir: str, output_dir: str, name:str):
    '''
    reads in all files, separates them in intro, summary and the single paragraphs, counts the lengths
    only considers the english versions of the files, prints if it fails to read a certain file and
    stores it in failed_files
    :param corpus_dir: directory of the corpus containing the text files
    :param output_dir: output directory where the pickled files of the lengths of each file, the paragraphs of each
    file and the failed_files are stored
    :return: the lengths of each file, the paragraphs of each file and the failed_files
    '''
    lengths = {}
    dict_paragraphs = {}
    failed_files = []
    for root, dirs, files in os.walk(corpus_dir):
        for file in files:
            print(file)
            with open(os.path.join(corpus_dir, file), 'r') as f:
                file = json.load(f)
                file_id = file.get('id')
                if file.get('plain_text'):
                    dict_paragraphs.update({file_id: {}})
                    lengths.update({file_id: {}})
                    plain_text = file.get('plain_text')

                    split_n = plain_text.split('\n')
                    split_sentences = [x.split('. ') for x in split_n]

                    common_sentences = [
                        'The syllabus constitutes no part of the opinion of the Court but has been prepared by the Reporter of Decisions for the convenience of the reader',
                        'Slip Opinion OCTOBER TERM 2009 1 Syllabus NOTE Where it is feasible a syllabus headnote will be released as is being done in connection with this case at the time the opinion is issued',
                        '(Slip Opinion)              OCTOBER TERM, 2016                                       1',
                        'Syllabus',
                        'NOTE: Where it is feasible, a syllabus (headnote) will be released, as is',
                        'being done in connection with this case, at the time the opinion is issued.',
                        'The syllabus constitutes no part of the opinion of the Court but has been',
                        'prepared by the Reporter of Decisions for the convenience of the reader.',
                        'See United States v', 'Detroit Timber & Lumber Co., 200 U', 'S', '321, 337.']
                    sentences = [item.strip() for sublist in split_sentences for item in sublist if
                                 item.strip() not in common_sentences]
                    lengths_para = []
                    paragraph_length = 0
                    paragraph = ''
                    i = 1
                    for sentence in sentences:
                        if sentence != '':
                            if paragraph_length < 200:
                                paragraph = paragraph + '. ' + sentence
                                sentence_length = count_words(sentence) if count_words(sentence) else 0
                                paragraph_length = paragraph_length + sentence_length
                            else:
                                paragraph = paragraph.strip('. ')
                                dict_paragraphs.get(file_id).update({str(i): paragraph})
                                lengths_para.append(paragraph_length)
                                paragraph_length = 0
                                paragraph = ''
                                i += 1
                    lengths.get(file_id).update({'intro': None, 'summary': None,
                                                 'lengths_paragraphs': lengths_para})
                else:
                    print('reading in of file {} doesnt work'.format(file))
                    failed_files.append(file)

    with open(os.path.join(output_dir, '{}_lengths.pickle'.format(name)), 'wb') as f:
            pickle.dump(lengths, f)
    with open(os.path.join(output_dir, '{}_paragraphs.pickle'.format(name)), 'wb') as f:
        pickle.dump(dict_paragraphs, f)

    return lengths, dict_paragraphs, failed_files


def jsonl_index_whole_doc(output_dir: str, dict_paragraphs: dict):
    """
    creates jsonl file for bm25 index and indexes the whole document with intro and summary as one sample
    :param output_dir:
    :param dict_paragraphs:
    :return:
    """
    with jsonlines.open(os.path.join(output_dir, 'corpus_whole_docs.jsonl'), mode='w') as writer:
        for key, values in dict_paragraphs.items():
            i = 0
            # values is also a dictionary with intro, summary and paragraphs
            whole_text = []
            for key2, values2 in values.items():
                if isinstance(values2, str):
                    whole_text.append(values2)
                elif isinstance(values2, dict):
                    for key3, values3 in values2.items():
                        whole_text.append(values3)
            writer.write({'id': '{}_{}'.format(key, i),
                          'contents': ' '.join(whole_text)})
            i += 1


def jsonl_index_para_separately(output_dir, dict_paragraphs):
    '''
    creates jsonl file with one sample containing one passage, if intro_summ = False then it only considers
    the paragraphs, if True then it also includes the intros and summaries as samples
    :param output_dir:
    :param dict_paragraphs:
    :param intro_summ:
    :return:
    '''
    with jsonlines.open(os.path.join(output_dir, 'corpus_separate_para.jsonl'), mode='w') as writer:
        for key, values in dict_paragraphs.items():
            i = 0
            for key2, values2 in values.items():
                if isinstance(values2, str):
                    writer.write({'id': '{}_{}'.format(key, i),
                                  'contents': values2})
                    i += 1
                elif isinstance(values2, dict):
                    if values2 != {}:
                        for key3, values3 in values2.items():
                            writer.write({'id': '{}_{}'.format(key, i),
                                          'contents': values3})
                            i += 1


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
                    if isinstance(values2, str):
                        values2.replace('\t', '')
                        values2.replace('\n', '')
                        whole_text.append(str(values2))
                    elif isinstance(values2, dict):
                        for key3, values3 in values2.items():
                            values3.replace('\t', '')
                            values3.replace('\n', '')
                            whole_text.append(str(values3))
                writer.writerow([' '.join(whole_text), ['{}_{}'.format(key, i)]])
                i += 1
            else:
                for key2, values2 in values.items():
                    if isinstance(values2, str):
                        writer.writerow([str(values2.replace(';', ',')), ['{}_{}'.format(key, i)]])
                        i += 1
                    elif isinstance(values2, dict):
                        for key3, values3 in values2.items():
                            writer.writerow([str(values3.replace(';', ',')), ['{}_{}'.format(key, i)]])
                            i += 1


def qa_for_train_test_file(dict_paragraphs_queries, output_dir, output_dir2):
    base_case_to_qa_file(dict_paragraphs_queries, output_dir, separate=False)
    base_case_to_qa_file(dict_paragraphs_queries, output_dir2, separate=True)


if __name__ == "__main__":
    corpus_dir = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/corpus'
    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/pickle_files'
    topics_dir = '/mnt/c/Users/salthamm/Documents/coding/ussc-caselaw-collection/airs2017-collection/topic'
    label_file = '/mnt/c/Users/salthamm/Documents/coding/ussc-caselaw-collection/airs2017-collection/qrel.txt'

    #lengths, dict_paragraphs, failed_files = read_in_para_lengths(corpus_dir, output_dir, 'corpus')
    #labels = preprocess_label_file(label_file)

    #with open(os.path.join(output_dir, 'corpus_lengths.pickle'), 'rb') as f:
    #    lengths = pickle.load(f)
    #with open(os.path.join(output_dir, 'corpus_paragraphs.pickle'), 'rb') as f:
    #    dict_paragraphs = pickle.load(f)

    # analyze corpus numbers
    #analyze_corpus_in_numbers(lengths, dict_paragraphs, labels, output_dir)
    #intro_often, summ_often, para_often = analyze_text_passages(dict_paragraphs, 100)

    lengths_queries, dict_paragraphs_queries, failed_files = read_in_para_lengths(topics_dir, output_dir, 'topics')
    #analyze_corpus_in_numbers(lengths_queries, dict_paragraphs_queries, labels, output_dir)

    # now to jsonl
    #jsonl_index_whole_doc(output_dir, dict_paragraphs)
    #jsonl_index_para_separately(output_dir, dict_paragraphs)

    #jsonl_index_whole_doc(output_dir, dict_paragraphs_queries)
    #jsonl_index_para_separately(output_dir, dict_paragraphs_queries)

    # jsonl to ctx for dpr
    jsonl_file = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/pickle_files/corpus_separate_para.jsonl'
    out_file = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/pickle_files/ctx/ctx_corpus_separate_para_seba.tsv'

    #with open(out_file, 'wt') as tsv_file:
    #    tsv_writer = csv.writer(tsv_file, delimiter='\t')
    #    tsv_writer.writerow(['id', 'text'])
    #    with jsonlines.open(jsonl_file) as reader:
    #        for obj in reader:
    #            #print(obj.get('contents').replace('"', ""))
    #            tsv_writer.writerow([obj.get('id'), obj.get('contents').replace('"', "")])

    #corpus_to_ctx_file(jsonl_file, out_file)

    # jsonl to qa files for dpr queries!
    #output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/pickle_files/qa/qa_whole_doc.csv'
    #output_dir2 = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/pickle_files/qa/qa_separate_para.csv'

    #qa_for_train_test_file(dict_paragraphs_queries, output_dir, output_dir2)

    # seba tsv file for queries
    out_file = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/pickle_files/qa/qa_separate_para.tsv'

    # from corpus jsonlines file to ctx_file format: .tsv file, id, text, title

    with open(out_file, 'wt') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t')
        tsv_writer.writerow(['id', 'text'])
        for key, values in dict_paragraphs_queries.items():
            #if key <= 32:
            i = 0
            for key2, values2 in values.items():
                if isinstance(values2, str):
                    tsv_writer.writerow(['{}_{}'.format(key, i), str(values2.replace(';', ','))])
                    i += 1
                elif isinstance(values2, dict):
                    for key3, values3 in values2.items():
                        tsv_writer.writerow(['{}_{}'.format(key, i), str(values3.replace(';', ','))])
                        i += 1



