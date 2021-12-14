import os
import ast
import argparse
from storing.elastic_util import ElasticSearch as elasticsearch


def bm25_search(file_path, output_dir, top_n, index, mode):
    #index = file_path.split('/')[-1].split('.jsonl')[0]
    #mode = output_dir.split('/')[-2]

    es = elasticsearch()
    trec_out = ""

    no_searches = 0
    with open(file_path) as fp:
        for line in fp:
            line_dict = ast.literal_eval(line)
            query_id = line_dict["id"]
            query_content = line_dict["contents"]

            bool_query = {
                "size": top_n,
                "query": {
                    "bool": {
                        "should": [
                            {"match": {'content': query_content}}
                        ],
                        "must_not": [
                            {"term": {'_id': query_id}}
                        ]

                        , "minimum_should_match": 0,
                        "boost": 1.0
                    }
                }
            }

            candidates = es.search(index=index, body=bool_query)
            rank = 1

            out = os.path.join(output_dir,'bm25_top{}_{}_{}.txt'.format(top_n, query_id, index))
            with open(out, "w", encoding="utf8") as out_file:
                for candidate in candidates['hits']['hits']:
                    document_id = candidate["_id"]
                    score = candidate["_score"]
                    line = "{query_id} Q0 {document_id} {rank} {score} STANDARD\n".format(query_id=query_id,
                                                                                          document_id=document_id,
                                                                                          rank=rank,
                                                                                          score=score)

                    trec_out += line
                    rank += 1
                    out_file.write(f'{document_id} {score:.5f}\n')

            no_searches += 1

            if no_searches % 100 == 0:
                print("{} searches done".format(no_searches))


    f_w = open(os.path.join('/'.join(output_dir.split('/')[:-1]), 'search_{}_{}.txt'.format(mode, index)),"w+")
    f_w.write(trec_out)
    f_w.close()


if __name__ == "__main__":
    """
    Example of usage:
        python -m rankers.bm25 \
            --file_path  /mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/val/separately_para_only.jsonl
            --output_dir  /mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/bm25/search/val/separately_para_only
            --top_n  1000
            --index clef-ip-whole-doc
            --mode whole-doc (oder separate-para)
    """
    parser = argparse.ArgumentParser('COLIEE2021 indexing')
    parser.add_argument('--file_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--top_n', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--mode', required=True)

    args = parser.parse_args()
    file_path = args.file_path
    output_dir = args.output_dir
    top_n = args.top_n
    index = args.index
    mode = args.mode

    bm25_search(file_path, output_dir, top_n, index, mode)