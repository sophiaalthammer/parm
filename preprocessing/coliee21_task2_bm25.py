import os
import re
import pickle
import json
import jsonlines
import matplotlib.pyplot as plt
import pytrec_eval
from pyserini.search import SimpleSearcher
from preprocessing.stat_corpus import lines_to_paragraphs, only_string_in_dict
from preprocessing.preprocessing_coliee_2021_task1 import only_english
from eval.eval_bm25 import read_label_file, read_run_whole_doc


def create_jsonl_candidates_for_pyserini(train_dir):

    list_dir = [x for x in os.walk(train_dir)]
    for sub_dir in list_dir[0][1]:
        with jsonlines.open(os.path.join(train_dir, sub_dir, 'candidates.jsonl'), mode='w') as writer:
            # read in all paragraphs with their names and then choose the relevant ones and sample irrelevant ones!
            list_sub_dir_paragraphs = [x for x in os.walk(os.path.join(train_dir, sub_dir, 'paragraphs'))]
            # paragraphs_text = {}
            for paragraph in list_sub_dir_paragraphs[0][2]:
                with open(os.path.join(train_dir, sub_dir, 'paragraphs', paragraph), 'r') as paragraph_file:
                    para_text = paragraph_file.read().splitlines()[1:]
                    writer.write({'id': '{}_{}'.format(sub_dir, paragraph.split('.')[0]),
                                  'contents': ' '.join([text.strip().replace('\n', '') for text in para_text])})


def search_index_coliee2021_task2(train_dir, out_dir, pickle_dir, top_n, mode, whole_doc=False):

    list_dir = [x for x in os.walk(train_dir)]
    with open(os.path.join(train_dir, 'failed_dirs.txt'),'w') as failed_dir:
        for sub_dir in list_dir[0][1]:
            try:
                # read in query text
                if whole_doc:
                    with open(os.path.join(train_dir, sub_dir, 'base_case.txt'), 'r') as entailed_fragment:
                        # here i should read in the base case as i do it also for docs
                        query_text = read_in_base_case(pickle_dir, entailed_fragment)
                else:
                    with open(os.path.join(train_dir, sub_dir, 'entailed_fragment.txt'), 'r') as entailed_fragment:
                        query_text_lines = entailed_fragment.read().splitlines()
                        query_text = ' '.join([text.strip().replace('\n', '') for text in query_text_lines])

                searcher = SimpleSearcher(os.path.join(train_dir, sub_dir, 'index'))
                searcher.set_bm25(0.9, 0.4)

                hits = searcher.search(query_text, top_n)

                # Print the first 50 hits:
                with open(os.path.join(out_dir, 'bm25_top{}_{}_whole_doc_{}.txt'.format(top_n, sub_dir, whole_doc)),"w",encoding="utf8") as out_file:
                    for hit in hits:
                        out_file.write(f'{hit.docid:55} {hit.score:.5f}\n')
            except:
                print('failed for {}'.format(sub_dir))
                failed_dir.write('{}\n'.format(sub_dir))


def search_coliee21_task2_for_failed_dirs(train_dir, out_dir, pickle_dir, top_n, mode, whole_doc=False):
    with open(os.path.join(train_dir, 'failed_dirs2.txt'), 'r') as failed_dir2:
        with open(os.path.join(train_dir, 'failed_dirs3.txt'), 'w') as failed_dir:
            for sub_dir in [x.strip('\n') for x in failed_dir2.readlines()]:
                try:
                    # read in query text
                    if whole_doc:
                        with open(os.path.join(train_dir, sub_dir, 'base_case.txt'), 'r') as entailed_fragment:
                            # here i should read in the base case as i do it also for docs
                            query_text = read_in_base_case(pickle_dir, entailed_fragment)
                            query_text = query_text[:6000]
                    else:
                        with open(os.path.join(train_dir, sub_dir, 'entailed_fragment.txt'), 'r') as entailed_fragment:
                            query_text_lines = entailed_fragment.read().splitlines()
                            query_text = ' '.join([text.strip().replace('\n', '') for text in query_text_lines])

                    searcher = SimpleSearcher(os.path.join(train_dir, sub_dir, 'index'))
                    searcher.set_bm25(0.9, 0.4)

                    hits = searcher.search(query_text, top_n)

                    # Print the first 50 hits:
                    with open(os.path.join(out_dir, 'bm25_top{}_{}_whole_doc_{}.txt'.format(top_n, sub_dir, whole_doc)),
                              "w", encoding="utf8") as out_file:
                        for hit in hits:
                            out_file.write(f'{hit.docid:55} {hit.score:.5f}\n')
                except:
                    print('failed for {}'.format(sub_dir))
                    failed_dir.write('{}\n'.format(sub_dir))


def read_in_base_case(pickle_dir, f):
    with open(os.path.join(pickle_dir, 'intro_text_often.pkl'), 'rb') as intro:
        intro_often = pickle.load(intro)
    with open(os.path.join(pickle_dir, 'summ_text_often.pkl'), 'rb') as summ:
        summ_often = pickle.load(summ)

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
    return ' '.join([x for key,x in paragraphs.items() if x])


def eval_ranking_bm25(label_file, bm25_folder, output_dir, output_file: str, scores='ranks'):
    qrels = read_label_file(label_file)
    run = read_run_whole_doc(bm25_folder, scores)

    measures = ranking_eval(qrels, run, output_dir, output_file)
    return measures


def ranking_eval(qrels, run, output_dir, output_file='eval_bm25_aggregate_overlap.txt', measures={'recall_1', 'recall_2', 'recall_3', 'recall_4', 'recall_5', 'recall_6', 'recall_7', 'recall_8', 'recall_9', 'recall_10',
                                                       'P_1', 'P_2', 'P_3', 'P_4', 'P_5', 'P_6', 'P_7', 'P_8', 'P_9', 'P_10'}):
    # trec eval
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures) #pytrec_eval.supported_measures)

    results = evaluator.evaluate(run)

    def print_line(measure, scope, value):
        print('{:25s}{:8s}{:.4f}'.format(measure, scope, value))

    def write_line(measure, scope, value):
        return '{:25s}{:8s}{:.4f}'.format(measure, scope, value)

    measures = {}
    for query_id, query_measures in sorted(results.items()):
        #for measure, value in sorted(query_measures.items()):
        #    print_line(measure, query_id, value)

        with open(os.path.join(output_dir, output_file), 'w') as output:
            for measure in sorted(query_measures.keys()):
                output.write(write_line(
                    measure,
                    'all',
                    pytrec_eval.compute_aggregated_measure(
                        measure,
                        [query_measures[measure]
                         for query_measures in results.values()])) + '\n')
                measures.update({measure: pytrec_eval.compute_aggregated_measure(measure,
                                [query_measures[measure]
                                 for query_measures in results.values()])})

    return measures

def plot_recall_precision(measures, eval_dir):
    plotting_data = {}
    for key, value in measures.items():
        if not plotting_data.get(key.split('_')[1]):
            plotting_data.update({key.split('_')[1]: [0, 0]})
        if 'P' in key:
            plotting_data.get(key.split('_')[1])[1] = value
        if 'recall' in key:
            plotting_data.get(key.split('_')[1])[0] = value

    #order them:
    desired_order_list = [int(key) for key, value in plotting_data.items()]
    desired_order_list.sort()
    desired_order_list = [str(x) for x in desired_order_list]
    plotting_data_sorted = {k: plotting_data[k] for k in desired_order_list}

    xs, ys = zip(*plotting_data_sorted.values())
    labels = plotting_data_sorted.keys()
    # display
    plt.figure(figsize=(10, 8))
    plt.xlabel('recall', fontsize=15)
    plt.ylabel('precision', fontsize=15)
    plt.scatter(xs, ys, marker='o')
    plt.plot(xs, ys)
    for label, x, y in zip(labels, xs, ys):
        plt.annotate(label, xy=(x, y))

    plt.savefig(os.path.join(eval_dir, 'prec_rec_plot.svg'))


if __name__ == "__main__":
    mode = ['train', True]
    train_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task2/{}'.format(mode[0])
    # create jsonl for indexing with pyserini
    #create_jsonl_candidates_for_pyserini(train_dir)

    # search index
    top_n = 50
    bm25_folder = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task2/bm25/search/{}/whole_doc_{}'.format(mode[0], mode[1])
    pickle_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/pickle_files'
    #search_index_coliee2021_task2(train_dir, bm25_folder, pickle_dir, top_n, mode, whole_doc=mode[1])
    #search_coliee21_task2_for_failed_dirs(train_dir, bm25_folder, pickle_dir, top_n, mode, whole_doc=mode[1])

    # evaluate bm25
    eval_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task2/bm25/eval/{}/whole_doc_{}'.format(mode[0], mode[1])
    output_file = 'eval_bm25_recall_{}_whole_doc_{}'.format(mode[0], mode[1])

    if mode[0] == 'train':
        label_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task2/task2_train_wo_val_labels_2021.json'
    elif mode [0] == 'val':
        label_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task2/task2_val_labels_2021.json'

    # evaluate ranking of bm25 as if it was for whole documents -> no aggregation needed
    measures = eval_ranking_bm25(label_file, bm25_folder, eval_dir, output_file)
    # plot the measures in a recall precision curve!
    plot_recall_precision(measures, eval_dir)










