# aicup-2023-nlp

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
