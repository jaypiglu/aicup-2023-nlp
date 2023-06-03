# aicup-2023-nlp

## Environment
python 3.9.12 \
torch 2.0.0 \
transformers 4.29.1 \
rank-bm25 0.2.2 \
ckip-transformers 0.3.4 \
wikipedia 1.4.0 \
hanlp 2.1.0b49 

## Dataset
從google drive id=1T6jpOtdf_i6XNYA6F_lqU4mRRh1xYPcl下載資料，GitHub放不下

## XNLI-finetune
直接跑script，可以參考https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification

## Document Retrieval
1. 以WikiAPI_retriever.py抽取10篇document
2. 以word_segmentation.py將claim斷詞，再以bm25_retriever.ipynb抽取100篇document
3. 以document_retrieval.ipynb混合WikiAPI及BM25的document

## Sentence Retrieval
1. 使用sent_retrieval_train.py進行訓練
2. 使用sent_retrieval_test.py進行預測

## Claim Verification
1. 使用claim_verification_train.py進行訓練
2. 使用claim_verification_test.py進行預測
