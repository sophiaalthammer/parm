import os
from preprocessing.caselaw_stat_corpus import preprocess_label_file, count_words
from analysis.caselaw_compare_bm25_dpr import read_in_run_from_pickle, remove_query_from_ranked_list, evaluate_weight
from analysis.diff_bm25_dpr import first_diff_analysis, write_case_lines, get_diff_query_ids, compare_overlap_rel


def read_in_qrels():
    # read in the label files
    label_file = '/mnt/c/Users/salthamm/Documents/coding/ussc-caselaw-collection/airs2017-collection/qrel.txt'

    qrels = preprocess_label_file(label_file)
    qrels_updated = {}
    for key, value in qrels.items():
        qrels_updated.update({key: {}})
        for val in value:
            qrels_updated.get(str(key)).update({str(val): 1})
    return qrels_updated


def write_diff_cases(query_diff_length_parm, query_diff_parm, output_dir, corpus_dir, query_file):
    diff_query_ids = get_diff_query_ids(query_diff_length_parm)

    query_dict = read_in_queries(query_file)

    for id in diff_query_ids:
        query_text = query_dict.get(id)
        bm25_text = {}
        dpr_text = {}
        intersect_text = {}
        if query_diff_parm.get(id).get('only_bm25'):
            for id_bm25 in query_diff_parm.get(id).get('only_bm25'):
                bm25_text.update({id_bm25: read_in_file(corpus_dir, id_bm25)})
        if query_diff_parm.get(id).get('only_dpr'):
            for id_dpr in query_diff_parm.get(id).get('only_dpr'):
                dpr_text.update({id_dpr: read_in_file(corpus_dir, id_dpr)})
        if query_diff_parm.get(id).get('intersect'):
            for id_int in query_diff_parm.get(id).get('intersect'):
                intersect_text.update({id_int: read_in_file(corpus_dir, id_int)})

        os.makedirs(os.path.join(output_dir, 'query_{}'.format(id)))
        with open(os.path.join(output_dir, 'query_{}'.format(id), 'query_{}.txt'.format(id)), 'w') as f:
            write_case_lines(query_text, f)
        for id_bm25, text in bm25_text.items():
            with open(os.path.join(output_dir, 'query_{}'.format(id), 'bm25_{}.txt'.format(id_bm25)), 'w') as f:
                write_case_lines(text, f)
        for id_dpr, text in dpr_text.items():
            with open(os.path.join(output_dir, 'query_{}'.format(id), 'dpr_{}.txt'.format(id_dpr)), 'w') as f:
                write_case_lines(text, f)
        for id_int, text in intersect_text.items():
            with open(os.path.join(output_dir, 'query_{}'.format(id), 'int_{}.txt'.format(id_int)), 'w') as f:
                write_case_lines(text, f)


def read_in_file(corpus_dir, id):
    dict_paragraphs = {}

    file = '{}.txt'.format(id)
    with open(os.path.join(corpus_dir, file), 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip('\n') is not ' ' and line.strip() is not '']
        paragraphs = {}
        paragraph = ''
        key = 'intro'
        for line in lines:
            if not line.split('.')[0].isdigit():
                paragraph = paragraph + ' ' + line
            else:
                # if paragraph is multiple times in document (for example in different languages)
                if key in paragraphs.keys():
                    para = paragraphs.get(key)
                    para.append(paragraph)
                    paragraphs.update({key: para})
                else:
                    paragraphs.update({key: [paragraph]})
                key = line.split('.')[0]
                paragraph = line
        if paragraphs:
            paragraphs = only_string_in_dict(paragraphs)
            dict_paragraphs.update({file.split('.')[0]: paragraphs})
            # print('lengths for file {} done'.format(file))
        else:
            print('reading in of file {} doesnt work'.format(file))
    return dict_paragraphs.get(id)


def read_in_queries(query_file):
    dict_paragraphs = {}
    with open(query_file, 'r') as f:
        lines = f.read().splitlines()
        for case in lines:
            query_id = case.split('||')[0]
            case = ' '.join(case.split('||')[1:])

            splitted_sentences = case.split('. ')
            paragraphs = {}
            line = ''
            i = 0
            lengths_docs = []
            for sentence in splitted_sentences:
                line = line + ' ' + sentence
                if len(line.split(' ')) > 200:
                    line_length = count_words(line)
                    lengths_docs.append(line_length)
                    paragraphs.update({str(i): line})
                    line = ''
                    i += 1
            # if there is one file which does not exceed the length of 200
            if not paragraphs:
                line_length = count_words(line)
                lengths_docs.append(line_length)
                paragraphs.update({'0': line})
            if paragraphs:
                dict_paragraphs.update({query_id: paragraphs})
            else:
                print(query_id)
    return dict_paragraphs


if __name__ == "__main__":
    mode = ['train', 'separate_para', 'overlap_ranks', 'legal_task1']
    # legalbert para
    #dpr_file_parm = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/dpr/legalbert/eval/run_dpr_aggregate_legalbert_parm_overlap_ranks.pickle'
    #dpr_file_doc = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/dpr/legalbert/eval/run_dpr_aggregate_legalbert_doc.pickle'
    # legalbert doc
    dpr_file_parm = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/dpr/legalbert_doc/eval/run_aggregated_train_vrrf.pickle'
    dpr_file_doc = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/dpr/legalbert_doc/eval/run_dpr_aggregate_firstp.pickle'
    bm25_file_parm = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/bm25/eval/run_bm25_aggregate2_parm_overlap_ranks.pickle'
    bm25_file_doc = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/bm25/eval/run_bm25_aggregate2_doc_overlap_ranks.pickle'
    output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/caselaw/bm25_dpr/legalbert_doc'

    # read in files for parm and doc
    dpr_dict_parm = read_in_run_from_pickle(dpr_file_parm)
    dpr_dict_parm = remove_query_from_ranked_list(dpr_dict_parm)

    dpr_dict_parm_new = {}
    for key, value in dpr_dict_parm.items():
        dpr_dict_parm_new.update({key.strip('id'):{}})
        for key2, value2 in value.items():
            dpr_dict_parm_new.get(key.strip('id')).update({key2.strip('id'): value2})
    dpr_dict_parm = dpr_dict_parm_new

    bm25_dict_parm = read_in_run_from_pickle(bm25_file_parm)
    bm25_dict_parm = remove_query_from_ranked_list(bm25_dict_parm)

    dpr_dict_doc = read_in_run_from_pickle(dpr_file_doc)
    dpr_dict_doc = remove_query_from_ranked_list(dpr_dict_doc)

    dpr_dict_parm_new = {}
    for key, value in dpr_dict_doc.items():
        dpr_dict_parm_new.update({key.strip('id'): {}})
        for key2, value2 in value.items():
            dpr_dict_parm_new.get(key.strip('id')).update({key2.strip('id'): value2})
    dpr_dict_doc = dpr_dict_parm_new

    bm25_dict_doc = read_in_run_from_pickle(bm25_file_doc)
    bm25_dict_doc = remove_query_from_ranked_list(bm25_dict_doc)

    # read in qrels
    qrels = read_in_qrels()

    qrels_len = []
    for key, value in qrels.items():
        qrels_len.append(len(value))
    print('number of total relevant docs is {}'.format(sum(qrels_len)))

    dpr_dict_parm_rel, bm25_dict_parm_rel, query_diff_parm, query_diff_length_parm = first_diff_analysis(dpr_dict_parm,
                                                                                                         bm25_dict_parm,
                                                                                                         qrels)
    compare_overlap_rel(dpr_dict_parm, bm25_dict_parm, qrels)

    dpr_dict_doc_rel, bm25_dict_doc_rel, query_diff_doc, query_diff_length_doc = first_diff_analysis(dpr_dict_doc,
                                                                                                     bm25_dict_doc,
                                                                                                     qrels)
    compare_overlap_rel(dpr_dict_doc, bm25_dict_doc, qrels)


    # compare doc and parm for bm25
    bm25_dict_rel_doc, bm25_dict_rel_parm, query_diff_bm25, query_diff_length_bm25 = first_diff_analysis(bm25_dict_doc,
                                                                                                         bm25_dict_parm,
                                                                                                         qrels)
    compare_overlap_rel(bm25_dict_rel_doc, bm25_dict_rel_parm, qrels)

    # compare doc and parm for dpr
    dpr_dict_rel_doc, dpr_dict_rel_parm, query_diff_dpr, query_diff_length_dpr = first_diff_analysis(dpr_dict_doc,
                                                                                                     dpr_dict_parm,
                                                                                                     qrels)
    compare_overlap_rel(dpr_dict_rel_doc, dpr_dict_rel_parm, qrels)