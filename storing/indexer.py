import os
import sys
import argparse
from storing.elastic_util import ElasticSearch as elasticsearch_util
import ast
import json
from distutils.util import strtobool


def get_docs(index_name, file_path, application_in_training, type):
    docs = []
    with open(file_path) as fp:
        for line in fp:
            if len(docs) % 500 == 0:
                print("{} docs parsed".format(len(docs)))
            line_dict = ast.literal_eval(line)
            id_ = line_dict["id"]
            contents = line_dict["contents"]
            doc = {
                "_index": index_name,
                "_id": id_,
                "application_in_training": application_in_training,  # train, test, cv
                "content": contents,
            }
            #if type == "para":
            #    doc.update({"para_num": line_dict["id"].split("_")[1]})
            docs.append(doc)
    return docs

def get_mapping(type):
    if type == "para":
        return json.loads(open("./storing/config/paragraph-level-mapping.json").read())
    else:
        return json.loads(open("./storing/config/whole-text-mapping.json").read())

def delete_index(index_name):
    es = elasticsearch_util()
    es.delete_index(index_name)


def indexing(type, index_name, file_path, application_in_training, replace_index):
    print("indexer run.. type: {}, index_name: {}, file_path: {}",
          type, index_name, file_path)
    docs = get_docs(index_name, file_path, application_in_training, type)
    es = elasticsearch_util()
    mapping = get_mapping(type)

    print("creating index mapping...")
    es.create_index(index_name, mapping, replace=replace_index)
    print("index mapping created !")

    print("indexing documents started...: ")
    es.index(index_name=index_name, documents=docs, is_bulk=True)
    print("all docs indexed :)")


if __name__ == "__main__":
    """
    Note: if your storage is almost full, run this command:
        curl -XPUT -H "Content-Type: application/json" http://localhost:9200/_cluster/settings -d '{ "transient": { "cluster.routing.allocation.disk.threshold_enabled": false } }'
        curl -XPUT -H "Content-Type: application/json" http://localhost:9200/_all/_settings -d '{"index.blocks.read_only_allow_delete": null}'
    
    Example of usage:
        python3.8 -m storing.indexer \
            --index_name separately_para_only \
            --type para \
            --application test \
            --replace_index false \
            --file_path  /mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/val/separately_para_only.jsonl
    """
    parser = argparse.ArgumentParser('COLIEE2021 indexing')
    parser.add_argument('--index_name', required=True)
    parser.add_argument('--application_in_training', choices=["train","test", "val", "corpus"])
    parser.add_argument('--replace_index', dest='replace_index', type=lambda x: bool(strtobool(x)), help="remove current index!")
    parser.add_argument('--type', choices=["whole", "para"], default='whole')
    parser.add_argument('--file_path', required=True)

    args = parser.parse_args()
    index_name = args.index_name
    type = args.type
    file_path = args.file_path
    application_in_training = args.application_in_training
    replace_index = args.replace_index
    indexing(type, index_name, file_path, application_in_training, replace_index)

    #index_name = 'separately_w_summ_intro'
    #delete_index(index_name)


