import os
import argparse
from pyserini.search import SimpleSearcher
from preprocessing.jsonlines_for_bm25_pyserini import read_in_docs


def search_bm25(index_dir: str, output_dir: str, dict_paragraphs: dict, top_n: int,split_doc: bool, intro_summ: bool, k_1=0.9, b_1=0.4):
    """
    searches the to the given documents in dict_paragraphs relevant documents in index_dir
    :param index_dir: directory of the index (indexed with pyserini)
    :param dict_paragraphs: dictionary with the query documents
    :param top_n: how many documents are retrieved from the corpus
    :param split_doc: if True, then for each paragraph of query document one search query and one result, if False then
    search for whole document
    :param intro_summ: if True, then the intros and summaries are contained in the search query, if False then they are
    removed
    :param k_1: for BM25 search
    :param b_1: for BM25 search
    :return:
    """
    with open(os.path.join(output_dir, 'failed_dirs.txt'), 'w') as failed_dir:
        for key, value in dict_paragraphs.items():
            # create query text, either with or without intro and summary
            query_text = []
            query_id = []
            i = 0
            for key2, value2 in value.items():
                if not intro_summ:
                    if key2 != 'intro' and key2 != 'Summary:':
                        if value2:
                            query_text.append(value2)
                            query_id.append(i)
                            i += 1
                else:
                    if value2:
                        query_text.append(value2)
                        query_id.append(i)
                        i += 1
            # set up searcher
            searcher = SimpleSearcher(index_dir)
            searcher.set_bm25(k_1, b_1)

            # query with query_text either as whole document or for each paragraph of the document
            if not split_doc:
                # here one could influence length of text, in case it gives memory errors
                query_text = ' '.join(query_text)
                try:
                    hits = searcher.search(query_text, top_n)
                    # Print the first top_n hits and write them in the out file
                    out = os.path.join(output_dir,
                                       'bm25_top{}_{}_intro_summ_{}.txt'.format(top_n, key, intro_summ))
                    with open(out, "w", encoding="utf8") as out_file:
                        for i in range(0, top_n):
                            out_file.write(f'{hits[i].docid:55} {hits[i].score:.5f}\n')
                except:
                    print('BM25 search failed for {}'.format(key))
                    failed_dir.write('{}\n'.format(key))

            else:
                assert len(query_text) == len(query_id)
                for j in range(len(query_text)):
                    try:
                        hits = searcher.search(query_text[j], top_n)
                        # Print the first top_n hits and write them in the out file
                        out = os.path.join(output_dir,
                                           'bm25_top{}_{}_para_{}_intro_summ_{}.txt'.format(top_n, key, query_id[j], intro_summ))
                        with open(out, "w", encoding="utf8") as out_file:
                            for i in range(0, top_n):
                                out_file.write(f'{hits[i].docid:55} {hits[i].score:.5f}\n')
                    except:
                        print('BM25 search failed for {} and for query paragraph with id {}'.format(key, query_id[j]))
                        failed_dir.write('{}_{}\n'.format(key, query_id[j]))


def search_for_failed_files(index_dir: str, output_dir: str, dict_paragraphs: dict, top_n: int,split_doc: bool, intro_summ: bool, shorten: int, k_1=0.9, b_1=0.4):

    with open(os.path.join(output_dir, 'failed_dirs2.txt'), 'r') as failed_dir_org:
        with open(os.path.join(output_dir, 'failed_dirs3.txt'), 'w') as failed_dir:
            keys = [x.strip() for x in failed_dir_org.readlines()]
            for key in keys:
                if split_doc:
                    j = int(key.split('_')[1])
                key = key.split('_')[0]
                value = dict_paragraphs.get(key)

                # create query text, either with or without intro and summary
                query_text = []
                query_id = []
                i = 0
                for key2, value2 in value.items():
                    if not intro_summ:
                        if key2 != 'intro' and key2 != 'Summary:':
                            if value2:
                                query_text.append(value2)
                                query_id.append(i)
                                i += 1
                    else:
                        if value2:
                            query_text.append(value2)
                            query_id.append(i)
                            i += 1

                # set up searcher
                searcher = SimpleSearcher(index_dir)
                searcher.set_bm25(k_1, b_1)

                # query with query_text either as whole document or for each paragraph of the document
                if not split_doc:
                    # here one could influence length of text, in case it gives memory errors
                    query_text = ' '.join(query_text)
                    # shorten the query_text
                    query_text = query_text[:shorten]
                    try:
                        hits = searcher.search(query_text, top_n)
                        # Print the first top_n hits and write them in the out file
                        out = os.path.join(output_dir,
                                           'bm25_top{}_{}_intro_summ_{}.txt'.format(top_n, key, intro_summ))
                        with open(out, "w", encoding="utf8") as out_file:
                            for i in range(0, top_n):
                                out_file.write(f'{hits[i].docid:55} {hits[i].score:.5f}\n')
                    except:
                        print('BM25 search failed for {}'.format(key))
                        failed_dir.write('{}\n'.format(key))

                else:
                    assert len(query_text) == len(query_id)
                    try:
                        hits = searcher.search(query_text[j][:shorten], top_n)
                        print(query_text[j])
                        # Print the first top_n hits and write them in the out file
                        out = os.path.join(output_dir,
                                           'bm25_top{}_{}_para_{}_intro_summ_{}.txt'.format(top_n, key, query_id[j], intro_summ))
                        with open(out, "w", encoding="utf8") as out_file:
                            for i in range(0, top_n):
                                out_file.write(f'{hits[i].docid:55} {hits[i].score:.5f}\n')
                    except:
                        print('BM25 search failed for {} and for query paragraph with id {}'.format(key, query_id[j]))
                        failed_dir.write('{}_{}\n'.format(key, query_id[j]))



if __name__ == "__main__":
    #
    # config
    #
    #parser = argparse.ArgumentParser()

    #parser.add_argument('--train-dir', action='store', dest='train_dir',
    #                    help='train directory location', required=True)

    #args = parser.parse_args()

    # first create index in bash!

    train_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2020/task1/base_case_val'
    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2020/task1/bm25/search_scores/val/separately_para_only'
    pickle_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2020/task1/pickle_files'
    index_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2020/task1/bm25/indices/separately_para_only'

    #
    # load directory structure
    #

    dict_paragraphs, failed_files = read_in_docs(train_dir, output_dir, pickle_dir, removal=True) # removal of non-informative text

    top_n = 1000
    search_bm25(index_dir, output_dir, dict_paragraphs, top_n, intro_summ=False, split_doc=True) # split the document and search for each paragraph, take also introduction and summary of search doucment?

    # with shortening to 8000 none fails for whole doc with summ and intro and wo
    search_for_failed_files(index_dir, output_dir, dict_paragraphs, top_n, intro_summ=False, split_doc=True, shorten=6000)


    # pyserini package: also tfidf weighting and retrieval! thats perfect for me!