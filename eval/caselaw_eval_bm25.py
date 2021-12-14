import os
import pytrec_eval
import seaborn as sns
import pickle
sns.set(color_codes=True, font_scale=1.2)
from preprocessing.caselaw_stat_corpus import preprocess_label_file
from eval.eval_bm25 import analyze_correlations_bet_para, aggregate_run_ranks_overlap, aggregate_run_mean_score, \
    aggregate_run_overlap, aggregate_run_interleave
from eval.eval_bm25_coliee2021 import aggregate_run_rrf


def eval_ranking_bm25(label_file, bm25_folder, output_dir, output_file: str, measurements, aggregation='interleave', scores='ranks'):

    if aggregation == 'overlap_scores':
        scores = 'scores'

    qrels = preprocess_label_file(label_file)
    qrels_updated = {}
    for key, value in qrels.items():
        qrels_updated.update({key: {}})
        for val in value:
            qrels_updated.get(str(key)).update({str(val): 1})

    if 'separate' in bm25_folder:
        print('i do separate')
        run = read_run_separate_aggregate(bm25_folder, aggregation, scores)
    else:
        print('i do whole doc')
        run = read_run_whole_doc(bm25_folder, scores)

    return run, qrels_updated


def read_run_whole_doc(bm25_folder: str, scores='ranks'):
    # geh in den bm25 folder, lies in dokument und query: dann dict {query: {top 1000}}
    run = {}
    for root, dirs, files in os.walk(bm25_folder):
        for file in files:
            with open(os.path.join(bm25_folder, file), 'r') as f:
                lines = f.readlines()
                lines_dict = {}
                for i in range(len(lines)):
                    if scores == 'scores':
                        lines_dict.update({lines[i].split(' ')[0].strip().strip('_0'): float(lines[i].split(' ')[-1].strip())})
                    else:
                        lines_dict.update({lines[i].split(' ')[0].strip().strip('_0'): float(len(lines) - i)})
                run.update({file.split('_')[2]: lines_dict})
    return run


def read_run_separate(bm25_folder: str, scores='ranks'):
    run = {}
    for root, dirs, files in os.walk(bm25_folder):
        for file in files:
            with open(os.path.join(bm25_folder, file), 'r') as f:
                lines = f.readlines()
                lines_dict = {}
                for i in range(len(lines)):
                    if scores == 'scores':
                        lines_dict.update({lines[i].split(' ')[0].strip().split('_')[0]: float(lines[i].split(' ')[-1].strip())})
                    else:
                        lines_dict.update({lines[i].split(' ')[0].strip().split('_')[0]: len(lines) - i})
                if run.get(file.split('_')[2]):
                    run.get(file.split('_')[2]).update({file.split('_')[3]: lines_dict})
                else:
                    run.update({file.split('_')[2]: {}})
                    run.get(file.split('_')[2]).update({file.split('_')[3]: lines_dict})
    return run


def read_run_separate_aggregate(bm25_folder: str, aggregation='interleave', scores='ranks'):
    # geh in den bm25 folder, lies in dokument und query: dann dict {query: {top 1000}}
    if aggregation == 'overlap_scores' or aggregation == 'mean_scores':
        scores = 'scores'

    run = read_run_separate(bm25_folder, scores)

    # now i need an aggregation function here, different choices
    if aggregation == 'overlap_docs':
        # now aggregate according to the overlap of the docs in the paragraphs!
        run_aggregated = aggregate_run_overlap(run)
    elif aggregation == 'interleave':
        run_aggregated = aggregate_run_interleave(run)
    elif aggregation == 'overlap_ranks':
        # now aggregate according to the overlap of the docs in the paragraphs!
        run_aggregated = aggregate_run_ranks_overlap(run)
    elif aggregation == 'overlap_scores':
        run_aggregated = aggregate_run_ranks_overlap(run)
    elif aggregation == 'mean_scores':
        run_aggregated = aggregate_run_mean_score(run)
    elif aggregation == 'rrf':
        run_aggregated = aggregate_run_rrf(run)
    if run_aggregated:
        return run_aggregated
    else:
        return run


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
    bm25_folder = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/bm25/search/{}'.format(mode[1])
    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/bm25/eval/{}/'.format(mode[1])

    if bm25_folder:
        if mode[0] == 'train':
            run, qrels = eval_ranking_bm25(label_file, bm25_folder, output_dir, measurements, 'eval_bm25_aggregate_{}.txt'.format(mode[2]),
                              aggregation=mode[2])
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

    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/bm25/eval/{}'.format(
        mode[1])
    output_file = 'eval_bm25_{}_{}_aggregate_{}.txt'.format(mode[0], mode[1], mode[2])

    ranking_eval2(qrels, run, output_dir, output_file, measurements)

    return run, qrels


if __name__ == "__main__":
    measurements = {'recall_100', 'recall_200', 'recall_300', 'recall_500', 'recall_1000', 'ndcg_cut_10', 'recip_rank'}

    label_file = '/mnt/c/Users/salthamm/Documents/coding/ussc-caselaw-collection/airs2017-collection/qrel.txt'

    #run2, qrels2 = eval_mode2(['train', 'whole_doc', 'overlap_docs'], measurements, label_file)
    #output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/bm25/eval'
    #with open(os.path.join(output_dir, 'run_bm25_aggregate2_doc_overlap_ranks.pickle'), 'wb') as f:
    #    pickle.dump(run2, f)
    run, qrel = eval_mode2(['train', 'separate_para', 'rrf'], measurements, label_file)
    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/bm25/eval'
    with open(os.path.join(output_dir, 'run_bm25_aggregate2_rrf_overlap_ranks.pickle'), 'wb') as f:
        pickle.dump(run, f)
    #eval_mode2(['train', 'separate_para', 'overlap_docs'], measurements, label_file)
    #eval_mode2(['train', 'separate_para', 'overlap_scores'], measurements, label_file)
    #eval_mode2(['train', 'separate_para', 'mean_scores'], measurements, label_file)
    #eval_mode2(['train', 'separate_para', 'interleave'], measurements, label_file)