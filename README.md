# Paragraph Aggregation Retrieval Model (PARM) for Dense Document-to-Document Retrieval

This repository contains the code for the paper **PARM: A Paragraph Aggregation Retrieval Model for Dense Document-to-Document Retrieval**
and is partly based on the [DPR Github repository](https://github.com/facebookresearch/DPR).
PARM is a Paragraph Aggregation Retrieval Model for dense document-to-document retrieval tasks,
which liberates dense passage retrieval models from their limited input lenght and
does retrieval on the paragraph-level.

We focus on the task of legal case retrieval and train and evaluate our models on
the [COLIEE 2021 data](https://sites.ualberta.ca/~rabelo/COLIEE2021/) and evaluate
our models on the [CaseLaw collection](https://github.com/ielab/ussc-caselaw-collection).

The dense retrieval models are trained on the COLIEE data and can be found [here](https://zenodo.org/record/5779380#.YbiieVnTU2w).
For training the dense retrieval model we utilize the [DPR Github repository](https://github.com/facebookresearch/DPR).

![PARM Workflow](documentation/parm_workflow.jpg?raw=true "Title")

If you use our models or code, please cite our work:

```
@inproceedings{althammer2022parm,
      title={Paragraph Aggregation Retrieval Model (PARM) for Dense Document-to-Document Retrieval}, 
      author={Althammer, Sophia and Hofstätter, Sebastian and Sertkan, Mete and Verberne, Suzan and Hanbury, Allan},
      year={2022},
      booktitle={Advances in Information Retrieval, 44rd European Conference on IR Research, ECIR 2022},
}
```


## Preprocessing

In order to train the dense retrieval models, the data needs to be preprocessed. For training and retrieval we
split up the documents into their paragraphs.

- ``./preprocessing/preprocess_finetune_data_dpr_task1.py``: preprocess the COLIEE Task 1 paragraph-level labels for training the DPR model

```bash
python preprocess_finetune_data_dpr_task1.py  --train-dir /home/data/coliee/task2/train --output-dir /home/data/coliee/task2/ouput 
```



## Model