import os
import seaborn as sns
import pickle
sns.set(color_codes=True, font_scale=1.2)
from collections import Counter
from eval.eval_bm25_coliee2021 import read_run_separate_aggregate
#from eval.eval_bm25 import read_run_separate_aggregate


def sort_write_trec_output(run, output_dir, mode):
    run_sorted = {}
    for query_id in run.keys():
        run_sorted.update(
            {query_id: {k: v for k, v in sorted(run.get(query_id).items(), key=lambda item: item[1], reverse=True)}})

    # then write in trec in aggregated version
    trec_out = ""
    for query_id in run_sorted.keys():
        rank = 1
        for document_id, score in run.get(query_id).items():
            line = "{query_id} Q0 {document_id} {rank} {score} STANDARD\n".format(query_id=query_id,
                                                                                  document_id=document_id,
                                                                                  rank=rank,
                                                                                  score=score)

            trec_out += line
            rank += 1

    f_w = open(os.path.join(output_dir, 'search_{}_{}_aggregation_{}.txt'.format(mode[0], mode[1], mode[2])), "w+")
    f_w.write(trec_out)
    f_w.close()

    return run_sorted


def aggregate_mode(mode):
    #bm25_folder = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2020/task1/bm25/search_scores/{}/{}'.format(mode[0], mode[1])
    #output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2020/task1/bm25/aggregate/{}/{}'.format(mode[0],mode[1])

    bm25_folder = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bm25/search/{}/separately_para_w_summ_intro/'.format(mode[0])
    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bm25/aggregate/{}/separately_para_w_summ_intro/'.format(mode[0])
    run = read_run_separate_aggregate(bm25_folder, mode[2])

    # sort dictionary by values
    #sort_write_trec_output(run, output_dir, mode)
    with open(os.path.join(output_dir, 'run_aggregated_{}_{}.pickle'.format(mode[0], mode[2])), 'wb') as f:
        pickle.dump(run, f)

    return run


if __name__ == "__main__":

    def aggregate_all_bm25(train_test):
        ## val ##
        # sep para: interleave
        #aggregate_mode([train_test, 'separately_para_only', 'interleave'])
        #aggregate_mode([train_test, 'separately_para_w_summ_intro', 'interleave'])

        # sep para: overlap docs
        aggregate_mode([train_test, 'separately_para_only', 'overlap_docs'])
        aggregate_mode([train_test, 'separately_para_w_summ_intro', 'overlap_docs'])

        # sep para: overlap ranks
        aggregate_mode([train_test, 'separately_para_only', 'overlap_ranks'])
        aggregate_mode([train_test, 'separately_para_w_summ_intro', 'overlap_ranks'])

        # sep para: overlap scores
        aggregate_mode([train_test, 'separately_para_only', 'overlap_scores'])
        aggregate_mode([train_test, 'separately_para_w_summ_intro', 'overlap_scores'])

        # sep para: mean scores
        #aggregate_mode([train_test, 'separately_para_only', 'mean_scores'])
        #aggregate_mode([train_test, 'separately_para_w_summ_intro', 'mean_scores'])


    aggregate_all_bm25('test')

    #run = aggregate_mode(['val', 'separately_para_w_summ_intro', 'overlap_ranks'])

    #from preprocessing.coliee21_task2_bm25 import ranking_eval
    #from eval.eval_bm25_coliee2021 import read_label_file

    #label_file_val = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/val/val_labels.json'
    #qrels = read_label_file(label_file_val)
    #output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bm25_dpr_legalbert/plot/val'
    #measurements = {'recall_100', 'recall_200', 'recall_300', 'recall_500', 'recall_1000', 'ndcg_cut_10', 'recip_rank'}

    #ranking_eval(qrels, run, output_dir, 'test_wo_aggregation_eval.txt', measurements)

    #run_sorted = sort_write_trec_output(run, output_dir, 'test_overlap_ranks')

    #ranking_eval(qrels, run_sorted, output_dir, 'test_run_sorted.txt', measurements)





