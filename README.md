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

## Training the dense retrieval model

The dense retrieval models need to be trained, either on the paragraph-level data of COLIEE Task2 or additionally on the document-level
data of COLIEE Task1

- ``./DPR/train_dense_encoder.py``: trains the dense bi-encoder (Step1)

```bash
python -m torch.distributed.launch --nproc_per_node=2 train_dense_encoder.py 
--max_grad_norm 2.0 
--encoder_model_type hf_bert 
--checkpoint_file_name --insert path to pretrained encoder checkpoint here if available-- 
--model_file  --insert path to pretrained chechpoint here if available-- 
--seed 12345 
--sequence_length 256 
--warmup_steps 1237 
--batch_size 22 
--do_lower_case 
--train_file --path to json train file-- 
--dev_file --path to json val file-- 
--output_dir --path to output directory--
--learning_rate 1e-05
--num_train_epochs 70
--dev_batch_size 22
--val_av_rank_start_epoch 60
--eval_per_epoch 1
--global_loss_buf_sz 250000
```

## Generate dense embeddings index with trained DPR model

- ``./DPR/generate_dense_embeddings.py``: encodes the corpus in the dense index (Step2)

```bash
python generate_dense_embeddings.py
--model_file --insert path to pretrained checkpoint here from Step1--
--pretrained_file  --insert path to pretrained chechpoint here from Step1--
--ctx_file --insert path to tsv file with documents in the corpus--
--out_file --insert path to output index--
--batch_size 750
```

## Search in the dense index

- ``./DPR/dense_retriever.py``: searches in the dense index the top n-docs (Step3)

```bash
python dense_retriever.py 
--model_file --insert path to pretrained checkpoint here from Step1--
--ctx_file --insert path to tsv file with documents in the corpus--
--qa_file --insert path to csv file with the queries--
--encoded_ctx_file --path to the dense index (.pkl format) from Step2--
--out_file --path to .json output file for search results--
--n-docs 1000
```


## Poolout dense vectors for aggregation step

First you need to get the dense embeddings for the query paragraphs:

- ``./DPR/get_question_tensors.py``: encodes the query paragraphs with the dense encoder checkpoint and stores the embeddings in the output file (Step4)

```bash
python get_question_tensors.py
--model_file --insert path to pretrained checkpoint here from Step1--
--qa_file --insert path to csv file with the queries--
--out_file --path to output file for output index--
```

Once you have the dense embeddings of the paragraphs in the index and of the questions, you do the vector-based aggregation step in PARM with VRRF (alternatively with Min, Max, Avg, Sum, VScores, VRanks)
and evaluate the aggregated results

- ``./representation_aggregation.py``: aggregates the run, stores and evaluates the aggregated run (Step5)

```bash
python representation_aggregation.py
--encoded_ctx_file --path to the encoded index (.pkl format) from Step2--
--encoded_qa_file  --path to the encoded queries (.pkl format) from Step4--
--output_top1000s --path to the top-n file (.json format) from Step3--
--label_file  --path to the label file (.json format)--
--aggregation_mode --choose from vrrf/vscores/vranks/sum/max/min/avg
--candidate_mode p_from_retrieved_list
--output_dir --path to output directory--
--output_file_name  --output file name--
```

## Preprocessing

Preprocess COLIEE Task 1 data for dense retrieval

- ``./preprocessing/preprocess_coliee_2021_task1.py``: preprocess the COLIEE Task 1 dataset by removing non-English text, removing non-informative summaries, removing tabs etc

Preprocess [CaseLaw collection](https://github.com/ielab/ussc-caselaw-collection)

- ``./preprocessing/caselaw_stat_corpus.py``: preprocess the CaseLaw collection 

### Preprocess data for training the dense retrieval model

In order to train the dense retrieval models, the data needs to be preprocessed. For training and retrieval we
split up the documents into their paragraphs.

- ``./preprocessing/preprocess_finetune_data_dpr_task1.py``: preprocess the COLIEE Task 1 document-level labels for training the DPR model

- ``./preprocessing/preprocess_finetune_data_dpr.py``: preprocess the COLIEE Task 2 paragraph-level labels for training the DPR model




