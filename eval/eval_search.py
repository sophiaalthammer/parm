import os
import argparse
import random
from statistics import mean
import xml.etree.ElementTree as ET
random.seed(42)

#
# config
#
parser = argparse.ArgumentParser()

parser.add_argument('--train-dir', action='store', dest='train_dir',
                    help='train directory location', required=True)
parser.add_argument('--test-gold-labels', action='store', dest='test_gold_labels',
                    help='test gold labels xml file', required=False)


args = parser.parse_args()

#train_dir = '/mnt/c/Users/sophi/Documents/phd/data/coliee2019/task1/task1_train'
#test_gold_labels = '/mnt/c/Users/sophi/Documents/phd/data/coliee2019/task1/task1_test_golden-labels.xml'

#
# load directory structure
#

list_dir = [x for x in os.walk(args.train_dir)]

if args.test_gold_labels:
    #
    # load gold labels as dictionary
    #

    tree = ET.parse(args.test_gold_labels)
    root = tree.getroot()

    gold_labels = {}
    for child in root:
        rank = child.find('cases_noticed').text
        rank = rank.split(',')
        gold_labels.update({child.attrib['id']: rank})


recall = []

for sub_dir in list_dir[0][1]:
    if args.test_gold_labels:
        doc_rel_id = gold_labels.get(sub_dir)
    else:
        # read in relevant document ids
        with open(os.path.join(args.train_dir, sub_dir, 'noticed_cases.txt'), 'r') as entailing_paragraphs:
            doc_rel_id = entailing_paragraphs.read().splitlines()

    with open(os.path.join(args.train_dir, sub_dir, 'bm25_top50.txt'), 'r') as top50:
        doc_bm25 = top50.read().splitlines()
        doc_bm25 = [doc.split('_')[1].strip() for doc in doc_bm25]

    recall.append(len(set(doc_rel_id) & set(doc_bm25))/len(doc_rel_id))


print(mean(recall))




# also add code for precision? and then F1-score?