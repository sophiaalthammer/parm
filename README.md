# Paragraph Aggregation Retrieval Model (PARM) for Dense Document-to-Document Retrieval

This repository contains the code for the paper **PARM: A Paragraph Aggregation Retrieval Model for Dense Document-to-Document Retrieval**
 of the paper [BERT-PLI: Modeling Paragraph-Level Interactions for Legal Case Retrieval](https://www.ijcai.org/Proceedings/2020/484) 
 and is based on the [BERT-PLI Github repository](https://github.com/ThuYShao/BERT-PLI-IJCAI2020).

We added the missing data [preprocessing](preprocessing) scripts as well as the script for [fine-tuning](finetune.py) the BERT model on binary classification, which
 is based on [HuggingFace' transformers library](https://github.com/huggingface/transformers). Furthermore
 we added scripts for the [evaluation](evaluation) with the [SciKitLearn classification report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html) as well as for the ranking evaluation 
 using the [pytrec_eval libary](https://github.com/cvangysel/pytrec_eval).



The open-sourced trained models can be found [here](https://zenodo.org/record/4088010).


## Outline

### Model