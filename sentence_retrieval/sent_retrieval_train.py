# %%
# built-in libs
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union

# third-party libs
import numpy as np
import pandas as pd
from pandarallel import pandarallel
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)


# local libs
from utils import (
    generate_evidence_to_wiki_pages_mapping,
    jsonl_dir_to_df,
    load_json,
    load_model,
    save_checkpoint,
    set_lr_scheduler,
)

pandarallel.initialize(progress_bar=True, verbose=0)

from torch.utils.data import Dataset
import opencc
CONVERTER_T2S = opencc.OpenCC("t2s.json")
CONVERTER_S2T = opencc.OpenCC("s2t.json")

NUM_DOC = 12

# %%
SEED = 42

TRAIN_DATA = load_json("../data/public_train.jsonl")
TEST_DATA = load_json("../data/public_test.jsonl")
DOC_DATA = load_json(f"../data/train_doc{NUM_DOC}.jsonl")

LABEL2ID: Dict[str, int] = {
    "supports": 0,
    "refutes": 1,
    "NOT ENOUGH INFO": 2,
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}

_y = [LABEL2ID[data["label"]] for data in TRAIN_DATA]
# GT means Ground Truth
TRAIN_GT, DEV_GT = train_test_split(
    DOC_DATA,
    test_size=0.05,
    random_state=SEED,
    shuffle=True,
    stratify=_y,
)

# %%
wiki_pages = jsonl_dir_to_df("../data/wiki-pages")
mapping = generate_evidence_to_wiki_pages_mapping(wiki_pages)
del wiki_pages

# %%
def evidence_macro_precision(
    instance: Dict,
    top_rows: pd.DataFrame,
) -> Tuple[float, float]:
    """Calculate precision for sentence retrieval
    This function is modified from fever-scorer.
    https://github.com/sheffieldnlp/fever-scorer/blob/master/src/fever/scorer.py

    Args:
        instance (dict): a row of the dev set (dev.jsonl) of test set (test.jsonl)
        top_rows (pd.DataFrame): our predictions with the top probabilities

        IMPORTANT!!!
        instance (dict) should have the key of `evidence`.
        top_rows (pd.DataFrame) should have a column `predicted_evidence`.

    Returns:
        Tuple[float, float]:
        [1]: relevant and retrieved (numerator of precision)
        [2]: retrieved (denominator of precision)
    """
    this_precision = 0.0
    this_precision_hits = 0.0

    # Return 0, 0 if label is not enough info since not enough info does not
    # contain any evidence.
    if instance["label"].upper() != "NOT ENOUGH INFO":
        # e[2] is the page title, e[3] is the sentence index
        all_evi = [[e[2], e[3]]
                   for eg in instance["evidence"]
                   for e in eg
                   if e[3] is not None]
        claim = instance["claim"]
        predicted_evidence = top_rows[top_rows["claim"] ==
                                      claim]["predicted_evidence"].tolist()

        for prediction in predicted_evidence:
            if prediction in all_evi:
                this_precision += 1.0
            this_precision_hits += 1.0

        return (this_precision /
                this_precision_hits) if this_precision_hits > 0 else 1.0, 1.0

    return 0.0, 0.0

# %%
def evidence_macro_recall(
    instance: Dict,
    top_rows: pd.DataFrame,
) -> Tuple[float, float]:
    """Calculate recall for sentence retrieval
    This function is modified from fever-scorer.
    https://github.com/sheffieldnlp/fever-scorer/blob/master/src/fever/scorer.py

    Args:
        instance (dict): a row of the dev set (dev.jsonl) of test set (test.jsonl)
        top_rows (pd.DataFrame): our predictions with the top probabilities

        IMPORTANT!!!
        instance (dict) should have the key of `evidence`.
        top_rows (pd.DataFrame) should have a column `predicted_evidence`.

    Returns:
        Tuple[float, float]:
        [1]: relevant and retrieved (numerator of recall)
        [2]: relevant (denominator of recall)
    """
    # We only want to score F1/Precision/Recall of recalled evidence for NEI claims
    if instance["label"].upper() != "NOT ENOUGH INFO":
        # If there's no evidence to predict, return 1
        if len(instance["evidence"]) == 0 or all(
            [len(eg) == 0 for eg in instance]):
            return 1.0, 1.0

        claim = instance["claim"]

        predicted_evidence = top_rows[top_rows["claim"] ==
                                      claim]["predicted_evidence"].tolist()

        for evidence_group in instance["evidence"]:
            evidence = [[e[2], e[3]] for e in evidence_group]
            if all([item in predicted_evidence for item in evidence]):
                # We only want to score complete groups of evidence. Incomplete
                # groups are worthless.
                return 1.0, 1.0
        return 0.0, 1.0
    return 0.0, 0.0

# %%
def evaluate_retrieval(
    probs: np.ndarray,
    df_evidences: pd.DataFrame,
    ground_truths: pd.DataFrame,
    top_n: int = 5,
    cal_scores: bool = True,
    save_name: str = None,
) -> Dict[str, float]:
    """Calculate the scores of sentence retrieval

    Args:
        probs (np.ndarray): probabilities of the candidate retrieved sentences
        df_evidences (pd.DataFrame): the candiate evidence sentences paired with claims
        ground_truths (pd.DataFrame): the loaded data of dev.jsonl or test.jsonl
        top_n (int, optional): the number of the retrieved sentences. Defaults to 2.

    Returns:
        Dict[str, float]: F1 score, precision, and recall
    """
    df_evidences["prob"] = [float(x) for x in probs]
    top_rows = (
        df_evidences.groupby("claim").apply(
        lambda x: x.nlargest(top_n, "prob"))
        .reset_index(drop=True)
    )

    if cal_scores:
        macro_precision = 0
        macro_precision_hits = 0
        macro_recall = 0
        macro_recall_hits = 0

        for i, instance in enumerate(ground_truths):
            macro_prec = evidence_macro_precision(instance, top_rows)
            macro_precision += macro_prec[0]
            macro_precision_hits += macro_prec[1]

            macro_rec = evidence_macro_recall(instance, top_rows)
            macro_recall += macro_rec[0]
            macro_recall_hits += macro_rec[1]

        pr = (macro_precision /
              macro_precision_hits) if macro_precision_hits > 0 else 1.0
        rec = (macro_recall /
               macro_recall_hits) if macro_recall_hits > 0 else 0.0
        f1 = 2.0 * pr * rec / (pr + rec)

    if save_name is not None:
        # write doc7_sent5 file
        with open(f"data/{save_name}", "w") as f:
            for instance in ground_truths:
                claim = instance["claim"]
                predicted_evidence = top_rows[
                    top_rows["claim"] == claim]["predicted_evidence"].tolist()
                instance["predicted_evidence"] = predicted_evidence
                f.write(json.dumps(instance, ensure_ascii=False) + "\n")

    if cal_scores:
        return {"F1 score": f1, "Precision": pr, "Recall": rec}

# %% [markdown]
# Inference script to get probabilites for the candidate evidence sentences

# %%
def get_predicted_probs(
    model: nn.Module,
    dataloader: Dataset,
    device: torch.device,
) -> np.ndarray:
    """Inference script to get probabilites for the candidate evidence sentences

    Args:
        model: the one from HuggingFace Transformers
        dataloader: devset or testset in torch dataloader

    Returns:
        np.ndarray: probabilites of the candidate evidence sentences
    """
    model.eval()
    probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            probs.extend(torch.softmax(logits, dim=1)[:, 1].tolist())

    return np.array(probs)

# %%
class CustomDataset(Dataset):
    
    def __init__(self, data_df, tokenizer, max_source_len, max_target_len):

        self.tokenizer = tokenizer
        self.data = data_df
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.source = self.tokenizer(
                self.data.source1.to_list(),
                self.data.source2.to_list(),
                padding='longest',
                truncation=True,
                max_length=self.max_source_len,
                return_tensors="pt"
            )
        
    def __len__(self):
        return len(self.source['input_ids'])

    def __getitem__(self, index):
        return {
            'input_ids': self.source['input_ids'][index],
            'attention_mask': self.source['attention_mask'][index],
            'labels': torch.tensor(self.data.target.to_list())[index]
        }

# %%
def pair_with_wiki_sentences(
    mapping: Dict[str, Dict[int, str]],
    df: pd.DataFrame,
    negative_ratio: float,
) -> pd.DataFrame:
    """Only for creating train sentences."""
    claims = []
    sentences = []
    labels = []

    # positive
    for i in range(len(df)):
        if df["label"].iloc[i] == "NOT ENOUGH INFO":
            continue

        claim = df["claim"].iloc[i]
        evidence_sets = df["evidence"].iloc[i]
        for evidence_set in evidence_sets:
            sents = []
            for evidence in evidence_set:
                # evidence[2] is the page title
                page = evidence[2].replace(" ", "_")
                # the only page with weird name
                if page == "臺灣海峽危機#第二次臺灣海峽危機（1958）":
                    continue
                # evidence[3] is in form of int however, mapping requires str
                sent_idx = str(evidence[3])
                sents.append(mapping[page][sent_idx])

            whole_evidence = " ".join(sents)

            claims.append(claim)
            sentences.append(whole_evidence)
            labels.append(1)

    # negative
    for i in range(len(df)):
        if df["label"].iloc[i] == "NOT ENOUGH INFO":
            continue
        claim = df["claim"].iloc[i]

        evidence_set = set([(evidence[2], evidence[3])
                            for evidences in df["evidence"][i]
                            for evidence in evidences])
        predicted_pages = df["predicted_pages"][i]
        for page in predicted_pages:
            page = page.replace(" ", "_")
            try:
                page_sent_id_pairs = [
                    (page, sent_idx) for sent_idx in mapping[page].keys()
                ]
            except KeyError:
                # print(f"{page} is not in our Wiki db.")
                continue

            for pair in page_sent_id_pairs:
                if pair in evidence_set:
                    continue
                text = mapping[page][pair[1]]
                # `np.random.rand(1) <= 0.05`: Control not to add too many negative samples
                if text != "" and np.random.rand(1) <= negative_ratio:
                    claims.append(claim)
                    sentences.append(text)
                    labels.append(0)

    return pd.DataFrame({"claim": claims, "text": sentences, "label": labels})


def pair_with_wiki_sentences_eval(
    mapping: Dict[str, Dict[int, str]],
    df: pd.DataFrame,
    is_testset: bool = False,
) -> pd.DataFrame:
    """Only for creating dev and test sentences."""
    claims = []
    sentences = []
    evidence = []
    predicted_evidence = []

    # negative
    for i in range(len(df)):
        # if df["label"].iloc[i] == "NOT ENOUGH INFO":
        #     continue
        claim = df["claim"].iloc[i]

        predicted_pages = df["predicted_pages"][i]
        for page in predicted_pages:
            page = page.replace(" ", "_")
            try:
                page_sent_id_pairs = [(page, k) for k in mapping[page]]
            except KeyError:
                # print(f"{page} is not in our Wiki db.")
                continue

            for page_name, sentence_id in page_sent_id_pairs:
                text = mapping[page][sentence_id]
                if text != "":
                    claims.append(claim)
                    sentences.append(text)
                    if not is_testset:
                        evidence.append(df["evidence"].iloc[i])
                    predicted_evidence.append([page_name, int(sentence_id)])

    return pd.DataFrame({
        "claim": claims,
        "text": sentences,
        "evidence": evidence if not is_testset else None,
        "predicted_evidence": predicted_evidence,
    })

# %%
train_df = pair_with_wiki_sentences(
    mapping,
    pd.DataFrame(TRAIN_GT),
    0.05,
)
counts = train_df["label"].value_counts()
print("Now using the following train data with 0 (Negative) and 1 (Positive)")
print(counts)

dev_df = pair_with_wiki_sentences_eval(mapping, pd.DataFrame(DEV_GT))

# %%
train_df['target'] = train_df['label']
dev_df['target'] = [0 for i in range(len(dev_df))]

train_df['source1'] = train_df['claim'].apply(lambda x: CONVERTER_T2S.convert(x))
train_df['source2'] = train_df['text'].apply(lambda x: CONVERTER_T2S.convert(x))
reverse_df = train_df.copy()
reverse_df['source2'] = train_df['source1']
reverse_df['source1'] = train_df['source2']
train_df =  pd.concat([train_df, reverse_df]).sample(frac=1)

dev_df['source1'] = dev_df['claim'].apply(lambda x: CONVERTER_T2S.convert(x))
dev_df['source2'] = dev_df['text'].apply(lambda x: CONVERTER_T2S.convert(x))

train_df = train_df.sample(frac=1)

# %%
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('/nfs/nas-6.1/wclu/AICUP/2023/lert/ckeckpoints/lert-base/checkpoint-12272', cache_dir = '/nfs/nas-6.1/wclu/cache', num_labels=2, ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained('/nfs/nas-6.1/wclu/AICUP/2023/lert/ckeckpoints/lert-base/checkpoint-12272', cache_dir = '/nfs/nas-6.1/wclu/cache')

# %%
train_dataset = CustomDataset(train_df, tokenizer, 512, 512)
dev_dataset = CustomDataset(dev_df, tokenizer, 512, 512)

# %%
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    probs = [x.softmax(dim=0)[1] for x in torch.tensor(predictions)]

    val_results = evaluate_retrieval(
        probs=np.array(probs),
        df_evidences=dev_df,
        ground_truths=DEV_GT,
        top_n=5,
    )
    return val_results

# %%
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# %%
name = 'lert-base-doc12-xnli_e2_bs64_reversed'

# %%
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir = f'checkpoints/{name}', 
    gradient_accumulation_steps = 4,
    evaluation_strategy = 'steps',
    eval_steps = 100,
    per_device_train_batch_size = 16, 
    per_device_eval_batch_size = 256,
    learning_rate = 2e-5,
    weight_decay = 0.01,                                 
    num_train_epochs = 5,                           
    warmup_ratio = 0.1,                         
    logging_dir = f'runs/{name}',
    logging_steps = 10,
    save_strategy = 'steps',
    save_steps = 100,
)

# %%
trainer = Trainer(
    model = model,                                       
    tokenizer = tokenizer,
    args = training_args,                                          
    train_dataset = train_dataset,                       
    eval_dataset = dev_dataset,                         
    compute_metrics = compute_metrics
)

# %%
trainer.train()


