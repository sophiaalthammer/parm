import os
import seaborn as sns
import pickle
sns.set(color_codes=True, font_scale=1.2)
from collections import Counter
from eval.eval_dpr_coliee2021 import read_run_separate_aggregate
from retrieval.bm25_aggregate_paragraphs import sort_write_trec_output
from preprocessing.coliee21_task2_dpr import read_run_whole_doc


def aggregate_mode(mode):
    pred_dir = '/mnt/c/Users/salthamm/Documents/coding/DPR/data/coliee2021_task1/{}/output/{}/{}_{}_top1000.json'.format(mode[3],mode[0], mode[0], mode[1])
    output_dir = '/mnt/c/Users/salthamm/Documents/coding/DPR/data/coliee2021_task1/{}/aggregate/{}'.format(mode[3], mode[0])

    run = read_run_separate_aggregate(pred_dir, mode[2])

    # sort dictionary by values
    #sort_write_trec_output(run, output_dir, mode)
    with open(os.path.join(output_dir, 'run_aggregated_{}_{}.pickle'.format(mode[0], mode[2])), 'wb') as f:
        pickle.dump(run, f)


def aggregate_coliee21_task2(mode):
    pred_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task2/dpr/output/train_wo_val/{}'.format(mode[0])
    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task2/dpr/aggregate/{}'.format(mode[0])

    run = read_run_whole_doc(pred_dir, mode[2])
    #sort_write_trec_output(run, output_dir, mode)
    with open(os.path.join(output_dir, 'run_aggregated_{}_{}.pickle'.format(mode[0], mode[2])), 'wb') as f:
        pickle.dump(run, f)


if __name__ == "__main__":


    def aggregate_all_bm25(train_test, model):
        ## val ##
        # sep para: interleave
        #aggregate_mode([train_test, 'separately_para_only', 'interleave', model])
        #aggregate_mode([train_test, 'separately_para_w_summ_intro', 'interleave', model])

        # sep para: overlap docs
        aggregate_mode([train_test, 'separate_para', 'overlap_docs', model])

        # sep para: overlap ranks
        aggregate_mode([train_test, 'separate_para', 'overlap_ranks', model])

        # sep para: overlap scores
        aggregate_mode([train_test, 'separate_para', 'overlap_scores', model])

        # sep para: mean scores
        #aggregate_mode([train_test, 'separately_para_only', 'mean_scores', model])
        #aggregate_mode([train_test, 'separately_para_w_summ_intro', 'mean_scores', model])


    #aggregate_all_bm25('train', 'legal_task2_dpr')
    #aggregate_all_bm25('train', 'standard_dpr')
    aggregate_all_bm25('test', 'legal_task2_dpr')
    #aggregate_all_bm25('val', 'standard_dpr')

    #aggregate_coliee21_task2(['val', 'something', 'scores'])

