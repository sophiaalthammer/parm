import os
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
import pickle


def lines_to_paragraphs(lines: list):
    '''
    creates a dictionary of paragraphs, the lines are accumulated to paragraphs according to their numbering of the lines
    :param lines:
    :return:
    '''
    paragraphs = {}
    paragraph = ''
    key = 'intro'
    for line in lines:
        if not (line == 'Summary:' or line.strip('[').strip(']').strip().isnumeric()):
            paragraph = paragraph + ' ' + line
        else:
            # if paragraph is multiple times in document (for example in different languages)
            if key in paragraphs.keys():
                para = paragraphs.get(key)
                para.append(paragraph)
                paragraphs.update({key: para})
            else:
                paragraphs.update({key:[paragraph]})
            key = line
            paragraph = ''
    # case for 002_200 if numbers of paragraphs are in same line as text
    if not paragraphs:
        for line in lines:
            if not line.strip('[').strip()[0].isdigit():
                paragraph = paragraph + ' ' + line
            else:
                if key == 'intro':
                    paragraphs.update({key: [paragraph]})
                paragraphs.update({line.split(']')[0] + ']': [']'.join(line.split(']')[1:])]})
    return paragraphs


def only_english(paragraphs: dict):
    # check intro where the english version is, is it the first one or the second one of the paragraphs?
    # but only if there are multiple options for the paratgraph
    freen = '[English language version follows French language version]'
    enfre = '[French language version follows English language version]'

    if enfre in paragraphs.get('intro')[0]:
        for key, value in paragraphs.items():
            if len(value) > 1:
                paragraphs.update({key: [value[0]]})
    elif freen in paragraphs.get('intro')[0]:
        for key, value in paragraphs.items():
            if len(value) > 1:
                paragraphs.update({key: [value[1]]})
    return paragraphs


def only_string_in_dict(paragraphs: dict):
    for key, value in paragraphs.items():
        paragraphs.update({key: value[0]})
    return paragraphs


def count_words(text: str):
    if text:
        return len(re.findall(r'\w+', text))
    else:
        return None

def count_doc(paragraphs: dict):
    # count intro
    text_intro = paragraphs.get('intro')
    no_intro = count_words(text_intro)

    # count summary
    text_summary = paragraphs.get('Summary:')
    if text_summary and not 'contains no summary' in text_summary:
        no_summ = count_words(text_summary)
    else:
        no_summ = None

    # paragraph lengths
    lengths = []
    for key, value in paragraphs.items():
        if key != 'intro' and key != 'Summary:':
            no_words = count_words(paragraphs[key])
            lengths.append(no_words)

    return no_intro, no_summ, lengths


def remove_duplicates_in_corpus(corpus_dir):
    # remove duplicates from corpus
    # first read in all text files and store them, then remove duplicates: read them in per line, just normal

    # corpus lines
    corpus_lines = read_folder(corpus_dir)

    # check duplicates
    files = list(corpus_lines.keys())
    duplicates = []
    for i in range(0, len(files)):
        file_text = corpus_lines.get(files[i])
        print(i)
        duplicate_list = [files[i]]
        # so that it does not check duplicates, if they have been already checked with another duplicate
        if not files[i] in list(itertools.chain.from_iterable(duplicates)):
            for j in range(i + 1, len(files)):
                print(j)
                if file_text == corpus_lines.get(files[j]):
                    duplicate_list.append(files[j])
        if len(duplicate_list) > 1:
            duplicates.append(duplicate_list)

    print('This is how many duplicate pairs(and maybe more than just 2 pairs) we have: {}'.format(len(duplicates)))
    print('This is how long on average the duplicate pairs are: {}'.format(np.mean([len(x) for x in duplicates])))

    # now remove duplicates from corpus, and also store which file is kept for its duplicates:
    # like {'076_015.txt': ['092_026.txt', '096_027.txt', '216_015.txt', '299_019.txt']}
    # just in case we remove relevant files, so that we still find which files is now replacing it!

    duplicates_removed = {}
    for duplicate_list in duplicates:
        for file in duplicate_list[1:]:
            os.remove(os.path.join(corpus_dir, file))
        duplicates_removed.update({duplicate_list[0]: duplicate_list[1:]})

    with open(os.path.join(output_dir, 'corpus_removed_duplicates.pkl'), 'wb') as f:
        pickle.dump(duplicates_removed, f)

    print('Removed {} duplicate files in total!'.format(
        len(list(itertools.chain.from_iterable(duplicates_removed.values())))))


def preprocess_label_file(output_dir, label_file):
    # now open the label file and the removed duplicates pickle file and
    # convert the label file in new format and duplicates

    with open(os.path.join(output_dir, 'corpus_removed_duplicates.pkl'), 'rb') as f:
        duplicates_removed = pickle.load(f)

    with open(label_file, 'rb') as f:
        labels = json.load(f)

    # other format of labels:
    labels_format = {}
    for key, values in labels.items():
        val_format = []
        for value in values:
            val_format.append('{}_{}'.format(key, value))
        labels_format.update({key: val_format})

    # turn mapping of duplicates removed around! key is the duplicate, value is the replacement
    duplicate_mapping = {}
    for key, values in duplicates_removed.items():
        for value in values:
            duplicate_mapping.update({value: key})

    # now replace duplicates which got replaced in the corpus with the file id which is still there
    labels_replaced = {}
    i = 0
    for key, values in labels_format.items():
        val_replaced = []
        for value in values:
            if duplicate_mapping.get(value):
                val_replaced.append(duplicate_mapping.get(value))
                print('replaced {} for {} with {}'.format(value, key, duplicate_mapping.get(value)))
                i += 1
            else:
                val_replaced.append(value)
        labels_replaced.update({key: val_replaced})
    print('replaced in total {} labels with their duplicate replacements'.format(i))

    with open(os.path.join(output_dir, 'labels_duplicates_removed.pkl'), 'wb') as f:
        pickle.dump(labels_replaced, f)

    return labels_replaced


def read_in_para_lengths(corpus_dir: str, output_dir: str):
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
            # file = '001_001.txt'
            with open(os.path.join(corpus_dir, file), 'r') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines if line.strip('\n') is not ' ' and line.strip() is not '']
                paragraphs = lines_to_paragraphs(lines)
                if paragraphs:
                    paragraphs = only_english(paragraphs)
                    paragraphs = only_string_in_dict(paragraphs)
                    # now analyze the intro, the summary and the length of the paragraphs
                    no_intro, no_summ, lengths_para = count_doc(paragraphs)
                    lengths.update({file.split('.')[0]: {'intro': no_intro, 'summary': no_summ,
                                                         'lengths_paragraphs': lengths_para}})
                    dict_paragraphs.update({file.split('.')[0]: paragraphs})
                    # print('lengths for file {} done'.format(file))
                else:
                    print('reading in of file {} doesnt work'.format(file))
                    failed_files.append(file)

    #with open(os.path.join(output_dir, 'base_case_lengths.pickle'), 'wb') as f:
    #    pickle.dump(lengths, f)
    #with open(os.path.join(output_dir, 'base_case_paragraphs.pickle'), 'wb') as f:
    #    pickle.dump(dict_paragraphs, f)
    #with open(os.path.join(output_dir, 'base_case_failed_files.pickle'), 'wb') as f:
    #    pickle.dump(failed_files, f)

    return lengths, dict_paragraphs, failed_files


def failed_files_in_labels(labels_replaced, failed_files):
    # check if the files which failed, are in the relevant ones - here we find that only one file is not read in although
    # it is relevant, reason: the files is empty, so we remove it form the corpus and the relevance assessments

    # reading in of file 350_170.txt failed, because it is empty
    # remove 350_170.txt from the labels, because this file is empty and also from the cleaned corpus!

    for key, values in labels_replaced.items():
        val_new = values
        for value in values:
            if value in failed_files:
                print(
                    'Attention we couldnt read in the relevant file {}, therefore we now remove it from the labels'.format(
                        value))
                val_new.remove(value)
                labels_replaced.update({key: val_new})
                print('updated dictionary to new pair {} for key {}'.format(val_new, key))

    with open(os.path.join(output_dir, 'labels_duplicates_removed_failed_files.pkl'), 'wb') as f:
        pickle.dump(labels_replaced, f)

    return labels_replaced


def count_duplicates_in_text(flipped:dict, threshold=100):
    """
    Counts how often a text passage is contained in different documents and returns the text passage which appear
    in more than 100 documents along with the number of occurences in different documents
    :param flipped: dictionary with text passages as key and list of documents which contain this text passage as values
    :return: list of numbers, how often this text passage is contained in different documents,
    list of text passages which are contained in more than 100 different documents,
    """
    paragraphs = []
    no_docs = []
    for para, docs in flipped.items():
        paragraphs.append(para)
        no_docs.append(len(docs))
    assert len(paragraphs) == len(no_docs)
    intro_often = []
    no_often = []
    for i in range(len(no_docs)):
        if no_docs[i] > threshold:
            intro_often.append(paragraphs[i])
            no_often.append(no_docs[i])
    return no_often, intro_often


def read_folder(folder_dir: str):
    corpus_lines = {}
    for root, dirs, files in os.walk(folder_dir):
        for file in files:
            with open(os.path.join(folder_dir, file), 'r') as f:
                lines = f.readlines()
                corpus_lines.update({file: lines})
    return corpus_lines


def analyze_text_passages(dict_paragraphs: dict, threshold=100):
    # first if some intros are the same
    flipped = {}
    for key, value in dict_paragraphs.items():
        if value.get('intro'):
            if value.get('intro') not in flipped:
                flipped[value.get('intro')] = [key]
            else:
                flipped[value.get('intro')].append(key)
    print('number of unique intros {}'.format(len(flipped)))
    no_often, intro_often = count_duplicates_in_text(flipped, threshold)
    print('This is how many intros are non-informative and will get removed: {}'.format(sum(no_often)))
    print('In total {} introductions make up for {} non-informative summaries in documents'.format(len(intro_often),
                                                                                               sum(no_often)))

    # now analyze the summaries
    flipped = {}
    for key, value in dict_paragraphs.items():
        if value.get('Summary:'):
            if value.get('Summary:') not in flipped:
                flipped[value.get('Summary:')] = [key]
            else:
                flipped[value.get('Summary:')].append(key)
    print('number of unique summary {}'.format(len(flipped)))
    no_often, summ_often = count_duplicates_in_text(flipped, threshold)
    print('This is how many summaries are non-informative and will get removed: {}'.format(sum(no_often)))
    print('In total {} summaries make up for {} non-informative summaries in documents'.format(len(summ_often), sum(no_often)))

    # analyze paragraphs
    flipped = {}
    for key, value in dict_paragraphs.items():
        for key2, value2 in value.items():
            if key2 != 'intro' and key2 != 'Summary:':
                if value2:
                    if value2 not in flipped:
                        flipped[value2] = [key]
                    else:
                        flipped[value2].append(key)
    print('number of unique paragraphs {}'.format(len(flipped)))
    no_often, para_often = count_duplicates_in_text(flipped, threshold)
    print('This is how many paragraphs are non-informative and will get removed: {}'.format(sum(no_often)))
    print('In total {} paragraphs make up for {} non-informative summaries in documents'.format(len(para_often),
                                                                                               sum(no_often)))

    # these text fragments will be removed from the files as they are not considered informative!
    # with open(os.path.join(output_dir, 'intro_text_often.pkl'), 'wb') as f:
    #    pickle.dump(intro_often, f)
    # with open(os.path.join(output_dir, 'summ_text_often.pkl'), 'wb') as f:
    #    pickle.dump(summ_often, f)
    # with open(os.path.join(output_dir, 'para_text_often.pkl'), 'wb') as f:
    #    pickle.dump(para_often, f)

    return intro_often, summ_often, para_often


def analyze_text_removal_from_base_case(dict_paragraphs: dict, output_dir):
    '''
    analyze how many of the parargaphs and which paragraphs would get removed, if removing non-informative
    # intros and summaries (attained from the corpus) from the base_cases
    :param bc_dict_paragraphs:
    :param output_dir:
    :return:
    '''
    with open(os.path.join(output_dir, 'intro_text_often.pkl'), 'rb') as f:
        intro_often = pickle.load(f)
    with open(os.path.join(output_dir, 'summ_text_often.pkl'), 'rb') as f:
        summ_often = pickle.load(f)

    flipped = {}
    for key, value in dict_paragraphs.items():
        if value.get('intro'):
            if value.get('intro') not in flipped:
                flipped[value.get('intro')] = [key]
            else:
                flipped[value.get('intro')].append(key)
    for intro in intro_often:
        print(intro)
        if flipped.get(intro):
            print('this is how many base_cases contain this intro and would get removed: {}'.format(len(flipped.get(intro))))
    print('number of unique intros {}'.format(len(flipped)))

    # now analyze the summaries
    flipped = {}
    for key, value in dict_paragraphs.items():
        if value.get('Summary:'):
            if value.get('Summary:') not in flipped:
                flipped[value.get('Summary:')] = [key]
            else:
                flipped[value.get('Summary:')].append(key)
    for summ in summ_often:
        print(summ)
        if flipped.get(summ):
            print('this is how many base_cases contain this summary and would get removed: {}'.format(len(flipped.get(summ))))
    print('number of unique summary {}'.format(len(flipped)))


def plot_hist(array: np.array, xaxis_title: str, title: str, output_dir: str):
    """
    plots a histogram for the given numpy array
    :param array: numpy array containing numbers
    :return: shows a histogram which displays the frequency for each number (for example year)
    """
    plt.figure()
    plot = sns.displot(array, binwidth=10, color="orange")  # , kde=False, hist_kws={"align": "left"}
    plt.axvline(x=np.mean(array), color='orange', linestyle='--')
    #plot.set(xticks=range(0, 1000, 100))
    #plot.set_xlim([-1000, 1000])
    plt.title(title)
    plt.ylabel("Frequency")
    plt.xlabel(xaxis_title)
    file_name = os.path.join(output_dir, 'plots/{0}_{1}_frequency.svg'.format(xaxis_title, title))
    #if not os.path.exists(os.path.dirname(file_name)):
    #    try:
    #        os.makedirs(os.path.dirname(file_name))
    #    except OSError as exc:  # Guard against race condition
    #        if exc.errno != errno.EEXIST:
    #            raise
    plt.savefig(file_name)
    #plt.show()


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
        avg_length.append(value.get('intro') if value.get('intro') else 0
                                                                        + value.get('summary') if value.get(
            'summary') else 0
                            + 0 if value.get('lengths_paragraphs') == [] else sum(
            [x for x in value.get('lengths_paragraphs') if x]))
    print('the documents have an average length of {}'.format(np.mean(avg_length)))
    print('the documents have an min length of {}'.format(np.min(avg_length)))
    print('the documents have an max length of {}'.format(np.max(avg_length)))
    # intros
    print(
        'number of documents with an intro {}'.format(sum([1 for key, value in lengths.items() if value.get('intro')])))
    print('the intros have an average length of {}'.format(
        np.mean([value.get('intro') for key, value in lengths.items() if value.get('intro')])))
    print('the shortest intro is {} words long'.format(
        np.min([value.get('intro') for key, value in lengths.items() if value.get('intro')])))
    print('the longest intro is {} words long'.format(
        np.max([value.get('intro') for key, value in lengths.items() if value.get('intro')])))
    plot_hist(np.array([value.get('intro') for key, value in lengths.items() if value.get('intro') and value.get('intro')<1000]),
                       'number of words', 'Introduction length distribution', output_dir)
    # summaries
    summ_len = []
    for key, value in dict_paragraphs.items():
        if value.get('Summary:'):
            summ_len.append(count_words(value.get('Summary:')))
    print('number of documents with a summary {}'.format(len(summ_len)))
    print('the summaries have an average length of {}'.format(np.mean(summ_len)))
    print('the shortest summary has {} words'.format(np.min(summ_len)))
    print('the longest summary has {} words'.format(np.max(summ_len)))
    plot_hist(np.array([x for x in summ_len if x<1000]), 'number of words', 'Summary length distribution', output_dir)
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
    #print('average number of relevant documents for test {}'.format(
    #    np.mean([len(value) for value in labels_test.values()])))


def check_duplicates_corpus_base_case(corpus_dir: str, base_case_dir: str):
    # check if base_cases are contained in corpus too?
    # first read in all text files and store them, then compare corpus with base_cases

    # corpus
    corpus_lines = read_folder(corpus_dir)

    # base_cases
    base_case_lines = read_folder(base_case_dir)

    # check duplicates of base_cases in corpus
    duplicates = {}
    for base_case in base_case_lines.keys():
        print(base_case)
        duplicate_list = []
        base_case_text = base_case_lines.get(base_case)
        for file in corpus_lines.keys():
            if base_case_text == corpus_lines.get(file):
                duplicate_list.append(file)
        if duplicate_list:
            duplicates.update({base_case: duplicate_list})

    print('This is how many base_cases also appear in the corpus: {}'.format(len(duplicates)))

    with open(os.path.join(output_dir, 'duplicates_base_case_corpus.pkl'), 'wb') as f:
        pickle.dump(duplicates, f)


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
    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2020/task1/pickle_files'
    label_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2020/task1/task1_train_2020_labels.json'
    label_file_test = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2020/task1/task1_test_2020_labels.json'
    base_case_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2020/task1/base_case_all'

    # remove_duplicates_in_corpus(corpus_dir)
    #lengths, dict_paragraphs, failed_files = read_in_para_lengths(corpus_dir, output_dir)

    #labels_replaced = preprocess_label_file(output_dir, label_file)
    # labels_replaced = failed_files_in_labels(labels_replaced, failed_files)

    # labels_replaced_test = preprocess_label_file(output_dir, label_file_test)
    # labels_replaced_test = failed_files_in_labels(labels_replaced_test, failed_files)

    with open(os.path.join(output_dir, 'corpus_lengths.pickle'), 'rb') as f:
        lengths = pickle.load(f)
    with open(os.path.join(output_dir, 'corpus_paragraphs.pickle'), 'rb') as f:
        dict_paragraphs = pickle.load(f)
    with open(os.path.join(output_dir, 'corpus_failed_files.pickle'), 'rb') as f:
        failed_files = pickle.load(f)
    with open(os.path.join(output_dir, 'labels_duplicates_removed.pkl'), 'rb') as f:
        labels_replaced = pickle.load(f)
    with open(os.path.join(output_dir, 'labels_test_duplicates_removed.pkl'), 'rb') as f:
        labels_replaced_test = pickle.load(f)

    # analyze corpus numbers
    analyze_corpus_in_numbers(lengths, dict_paragraphs, labels_replaced, output_dir)

    # analyze the text
    #intro_often, summ_often, para_often = analyze_text_passages(dict_paragraphs, 100)

    # analyze the base cases
    #check_duplicates_corpus_base_case(corpus_dir, base_case_dir)
    #bc_lengths, bc_dict_para, bc_failed_files = read_in_para_lengths(base_case_dir, output_dir)

    #analyze_corpus_in_numbers(bc_lengths, bc_dict_para, labels_replaced, output_dir)
    #analyze_text_passages(bc_dict_para, 50)

    # how many of the non-informative text of the corpus are in the base_cases?
    #analyze_text_removal_from_base_case(bc_dict_para, output_dir)







