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
            if top_k is not None and int(rank_number) > top_k: continue
            if q_id == "para": continue #one query id is 'para' in the ranking for test run! I will talk about that query with Sophia later!
            if q_id not in ranking_dict:
                ranking_dict[q_id] = []
            ranking_dict[q_id].append((c_id, rank_number))
    return ranking_dict

def main(ranking_top_k_path, output_path, jsonl_corpus_path):
    #loading corpus as a json, structrue: { "d_id": [para_0_text, .... para_N_text]}
    json_corpus = load_json_corpus(jsonl_corpus_path)

    #structure: {q_id: [rank1_id, ... rank_k_id}
    top_k = 500
    with jsonlines.open(output_path, mode='w') as writer:
        first_stage_ranking_dict = load_ranking(ranking_top_k_path, top_k=None)
        for query_id, retrieved_docs in first_stage_ranking_dict.items():
            for did, rank_number in retrieved_docs:
                query_text = json_corpus[query_id] #list of paras
                guid = '{}_{}'.format(query_id, did)
                para_text = json_corpus[did]

                rank_number = int(rank_number)
                if rank_number<=top_k:
                    out_ = {'guid': guid,
                            'q_paras': query_text,
                            'c_paras': para_text,
                            'label': 1 }
                    writer.write(out_)

                elif rank_number>top_k:
                    out_ = {'guid': guid,
                            'q_paras': query_text,
                            'c_paras': para_text,
                            'label': 0}
                    # writer.write(out_)

if __name__ == "__main__":
    """
    Example of usage:
        python3.8 -m preprocessing.coliee21_task1_json_lines_test \
            --ranking_top_k_path /home/arian/phd/code/coliee2021/data/task1/bm25/search/test/search_test_separately_para_w_summ_intro_aggregation_overlap_ranks.txt \
            --output_path /home/arian/phd/code/coliee2021/data/task1/doc_para/test/input_fo_doc_para_formatter.json \
            --jsonl_corpus_path /home/arian/phd/code/coliee2021/data/task1/corpus_jsonl/separately_para_w_summ_intro/separately_para_w_summ_intro.jsonl \
    """
    parser = argparse.ArgumentParser('COLIEE2021 indexing')
    parser.add_argument('--ranking_top_k_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--jsonl_corpus_path', default= "/home/arian/phd/code/coliee2021/data/task1/corpus_jsonl/separately_para_w_summ_intro/separately_para_w_summ_intro.jsonl")

    args = parser.parse_args()
    ranking_top_k_path = args.ranking_top_k_path
    jsonl_corpus_path = args.jsonl_corpus_path
    output_path = args.output_path
    main(ranking_top_k_path, output_path, jsonl_corpus_path)