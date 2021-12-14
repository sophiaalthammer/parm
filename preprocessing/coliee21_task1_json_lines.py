import argparse
import random
import jsonlines
import ast
import json
random.seed(42)

def load_json_corpus(path):
    with open(path) as fp:
        cnt = 0
        json_corpus = dict()
        qid_text = dict()
        cur_id = ""
        for line in fp:
            cnt += 1
            if cnt % 10000 == 0:
                print("{} lines parsed".format(cnt))
            line_dict = ast.literal_eval(line)
            id_ = line_dict["id"]
            if "097104" in id_:
                print(id_)
                print(id_.split("_")[0])
            id_ = id_.split("_")[0]
            contents = line_dict["contents"]
            if cur_id == "":
                cur_id = id_
                qid_text[id_] = [contents]  # contents: one paragraph of a doc

            elif cur_id == id_:
                qid_text[id_].append(contents)

            elif cur_id != id_:
                json_corpus.update(qid_text)

                qid_text = dict()
                cur_id = id_
                qid_text[id_] = [contents]  # contents: one paragraph of a doc

        json_corpus.update(qid_text) #for last one!
    return json_corpus

def load_ranking(qrel_path, top_k=500):
    ranking_dict = dict()
    with open(qrel_path, 'r') as ranking_file:
        ranking_topk_lines = ranking_file.read().splitlines()
        for ranking_line in ranking_topk_lines:
            q_id, run_name, c_id, rank_number, score_rel, run_mode = ranking_line.split(" ")
            if int(rank_number) > top_k: continue
            if q_id not in ranking_dict:
                ranking_dict[q_id] = []
            ranking_dict[q_id].append(c_id)
    return ranking_dict

def main(ranking_top_k_path, mode, output_path, jsonl_corpus_path, labels_path):
    #loading corpus as a json, structrue: { "d_id": [para_0_text, .... para_N_text]}
    json_corpus = load_json_corpus(jsonl_corpus_path)

    #structure: {q_id: [rank1_id, ... rank_k_id}
    first_stage_ranking_dict = load_ranking(ranking_top_k_path, top_k=500)

    #note: do re-sampling in training phase!
    """ final steps to create output! """
    queries_train = json.loads(open(labels_path, "r").read())
    with jsonlines.open(output_path, mode='w') as writer:
        for query, doc_rel_list in queries_train.items():
                query_id = query.replace(".txt", "").strip()
                query_text = json_corpus[query_id] #list of paras
                try:
                    top_k_docs = first_stage_ranking_dict[query_id]
                except Exception as exception:
                    print(repr(exception), "\t wasn't exist in first stage ranking! reason, if mode==train/val: because that id belongs to val/train set; if mode ==test: becasue that id does not belong to top-k candidates")
                    continue

                pos_docs_id = [doc_rel_id.replace(".txt", "") for doc_rel_id in doc_rel_list]

                if mode == "train":
                    neg_docs_id = random.sample([did for did in top_k_docs if did not in pos_docs_id], len(pos_docs_id))
                elif mode == "val":
                    neg_docs_id = [did for did in top_k_docs if did not in pos_docs_id]

                elif mode == "test":
                    neg_docs_id = [did for did in top_k_docs if did not in pos_docs_id]

                selected_docs = pos_docs_id + neg_docs_id

                for did in selected_docs:
                    guid = '{}_{}'.format(query_id, did)
                    para_text = json_corpus[did]
                    out_ = {'guid': guid,
                     'q_paras': query_text,
                     'c_paras': para_text,
                     'label': 1 if did in pos_docs_id else 0}
                    writer.write(out_)

if __name__ == "__main__":
    """
    Example of usage:
        python3.8 -m preprocessing.coliee21_task1_json_lines \
            --ranking_top_k_path /home/arian/phd/code/coliee2021/data/task1/bm25/search/train/search_train_separately_para_w_summ_intro_aggregation_overlap_ranks.txt \
            --output_path /home/arian/phd/code/coliee2021/data/task1/doc_para/train/input_fo_doc_para_formatter.json \
            --mode  train \
            --jsonl_corpus_path /home/arian/phd/code/coliee2021/data/task1/corpus_jsonl/separately_para_w_summ_intro/separately_para_w_summ_intro.jsonl \
            --labels_path /home/arian/phd/datasets/COLIEE-2021/train_labels.json
            
            
        python3.8 -m preprocessing.coliee21_task1_json_lines \
            --ranking_top_k_path /home/arian/phd/code/coliee2021/data/task1/bm25/search/val/search_val_separately_para_only_aggregation_overlap_ranks.txt \
            --output_path /home/arian/phd/code/coliee2021/data/task1/doc_para/val/input_fo_doc_para_formatter.json \
            --mode  val \
            --jsonl_corpus_path /home/arian/phd/code/coliee2021/data/task1/corpus_jsonl/separately_para_w_summ_intro/separately_para_w_summ_intro.jsonl \
            --labels_path /home/arian/phd/datasets/COLIEE-2021/train_labels.json
    """
    parser = argparse.ArgumentParser('COLIEE2021 indexing')
    parser.add_argument('--ranking_top_k_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--mode', choices=["train", "test", "val"])
    parser.add_argument('--jsonl_corpus_path', default= "/home/arian/phd/code/coliee2021/data/task1/corpus_jsonl/separately_para_w_summ_intro/separately_para_w_summ_intro.jsonl")
    parser.add_argument('--labels_path', default='/home/arian/phd/datasets/COLIEE-2021/train_labels.json')

    args = parser.parse_args()
    ranking_top_k_path = args.ranking_top_k_path
    mode = args.mode
    output_path = args.output_path
    jsonl_corpus_path = args.jsonl_corpus_path
    labels_path = args.labels_path
    main(ranking_top_k_path, mode, output_path, jsonl_corpus_path, labels_path)