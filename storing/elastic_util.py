import elasticsearch
from elasticsearch import helpers

class ElasticSearch:
    def __init__(self):
        # configure elasticsearch
        ELASTI_HOST = "localhost"
        ELASTIC_PORT = "9200"

        config = {
            ELASTI_HOST: ELASTIC_PORT
        }
        self.es = elasticsearch.Elasticsearch([config, ], timeout=300)

    def create_index(self, name, mapping, replace=False):
        if replace:
            self.delete_index(name)
        print("creating index, name: ", name)
        try:
            self.es.indices.create(index=name, body=mapping)
        except Exception as e:
                print("\nERROR:", e)
        print("index created successfully, index name: " + name)

    def delete_index(self, name):
        print("deleting index, name: ", name)
        self.es.indices.delete(index=name, ignore=[400, 404])
        print("index deleted successfully, index name: " + name)

    def index(self, documents, index_name, is_bulk=False):
        if is_bulk:
            try:
                # make the bulk call, and get a response
                response = helpers.bulk(self.es, documents)  # chunk_size=1000, request_timeout=200
                print("\nRESPONSE:", response)
            except Exception as e:
                print("\nERROR:", e)

    def search(self, index, body):
        try:
            # make the bulk call, and get a response
            return self.es.search(index=index, body=body)
        except Exception as e:
            print("\nERROR:", e)
    def termvectors(self, index, body, id):
        try:
            # make the bulk call, and get a response
            return self.es.termvectors(index=index, body=body, id=id)
        except Exception as e:
            print("\nERROR:", e)