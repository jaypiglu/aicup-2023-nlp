# %%
import pickle
from pathlib import Path
from typing import Dict, Tuple
import json

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from tqdm.auto import tqdm

import torch
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)

from utils import (
    generate_evidence_to_wiki_pages_mapping,
    jsonl_dir_to_df,
    load_json,
    load_model,
    save_checkpoint,
    set_lr_scheduler,
)
from torch.utils.data import Dataset
import opencc
CONVERTER_T2S = opencc.OpenCC("t2s.json")
CONVERTER_S2T = opencc.OpenCC("s2t.json")

pandarallel.initialize(progress_bar=True, verbose=0)

# %%
wiki_pages = jsonl_dir_to_df("../data/wiki-pages")
mapping = generate_evidence_to_wiki_pages_mapping(wiki_pages,)
del wiki_pages

# %%
def run_evaluation(model: torch.nn.Module, dataloader: DataLoader, device):
    model.eval()

    loss = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            y_true.extend(batch["labels"].tolist())

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss += outputs.loss.item()
            logits = outputs.logits
            y_pred.extend(torch.argmax(logits, dim=1).tolist())

    acc = accuracy_score(y_true, y_pred)

    return {"val_loss": loss / len(dataloader), "val_acc": acc}

# %% [markdown]
# Prediction

# %%
def run_predict(model: torch.nn.Module, test_dl: DataLoader, device) -> list:
    model.eval()

    preds = []
    for batch in tqdm(test_dl,
                      total=len(test_dl),
                      leave=False,
                      desc="Predicting"):
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(**batch).logits
        pred = torch.argmax(pred, dim=1)
        preds.extend(pred.tolist())
    return preds

# %% [markdown]
# ### Main function

# %%
def join_with_topk_evidence(
    df: pd.DataFrame,
    mapping: dict,
    mode: str = "train",
    topk: int = 5,
) -> pd.DataFrame:
    """join_with_topk_evidence join the dataset with topk evidence.

    Note:
        After extraction, the dataset will be like this:
               id     label         claim                           evidence            evidence_list
        0    4604  supports       高行健...     [[[3393, 3552, 高行健, 0], [...  [高行健 （ ）江西赣州出...
        ..    ...       ...            ...                                ...                     ...
        945  2095  supports       美國總...  [[[1879, 2032, 吉米·卡特, 16], [...  [卸任后 ， 卡特積極參與...
        停各种战争及人質危機的斡旋工作 ， 反对美国小布什政府攻打伊拉克...

        [946 rows x 5 columns]

    Args:
        df (pd.DataFrame): The dataset with evidence.
        wiki_pages (pd.DataFrame): The wiki pages dataframe
        topk (int, optional): The topk evidence. Defaults to 5.
        cache(Union[Path, str], optional): The cache file path. Defaults to None.
            If cache is None, return the result directly.

    Returns:
        pd.DataFrame: The dataset with topk evidence_list.
            The `evidence_list` column will be: List[str]
    """

    # format evidence column to List[List[Tuple[str, str, str, str]]]
    if "evidence" in df.columns:
        df["evidence"] = df["evidence"].parallel_map(
            lambda x: [[x]] if not isinstance(x[0], list) else [x]
            if not isinstance(x[0][0], list) else x)

    print(f"Extracting evidence_list for the {mode} mode ...")
    if mode == "eval":
        # extract evidence
        df["evidence_list"] = df["predicted_evidence"].parallel_map(lambda x: [
            mapping.get(evi_id, {}).get(str(evi_idx), "")
            for evi_id, evi_idx in x  # for each evidence list
        ][:topk] if isinstance(x, list) else [])
        print(df["evidence_list"][:5])
    else:
        # extract evidence
        df["evidence_list"] = df["evidence"].parallel_map(lambda x: [
            " ".join([  # join evidence
                mapping.get(evi_id, {}).get(str(evi_idx), "")
                for _, _, evi_id, evi_idx in evi_list
            ]) if isinstance(evi_list, list) else ""
            for evi_list in x  # for each evidence list
        ][:1] if isinstance(x, list) else [])

    return df

# %%
NUM_DOC = 12
EVIDENCE_TOPK = 5
model_name = 'lert-large-doc12-xnli_e1_700'
#model_name = 'lert-base-doc12-xnli_e2_500'
#model_name = 'lert-base-doc12-xnli_e2_bs64_reversed'


LABEL2ID: Dict[str, int] = {
    "supports": 0,
    "NOT ENOUGH INFO": 1,
    "refutes": 2
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}

TRAIN_DATA = load_json(f"../data/train_doc{NUM_DOC}sent5_{model_name}.jsonl")
TRAIN_PKL_FILE = Path(f"../data/train_doc{NUM_DOC}sent5_{model_name}.pkl")
if not TRAIN_PKL_FILE.exists():
    train_df = join_with_topk_evidence(
        pd.DataFrame(TRAIN_DATA),
        mapping,
        mode="eval",
        topk=EVIDENCE_TOPK,
    )
    train_df.to_pickle(TRAIN_PKL_FILE, protocol=4)
else:
    with open(TRAIN_PKL_FILE, "rb") as f:
        train_df = pickle.load(f)

DEV_DATA = load_json(f"../data/dev_doc{NUM_DOC}sent5_{model_name}.jsonl")
DEV_PKL_FILE = Path(f"../data/dev_doc{NUM_DOC}sent5_{model_name}.pkl")
if not DEV_PKL_FILE.exists():
    dev_df = join_with_topk_evidence(
        pd.DataFrame(DEV_DATA),
        mapping,
        mode="eval",
        topk=EVIDENCE_TOPK,
    )
    dev_df.to_pickle(DEV_PKL_FILE, protocol=4)
else:
    with open(DEV_PKL_FILE, "rb") as f:
        dev_df = pickle.load(f)


# %%
class CustomDataset(Dataset):
    
    def __init__(self, data_df, tokenizer, max_source_len, max_target_len):

        self.tokenizer = tokenizer
        self.data = data_df
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.source = self.tokenizer(
                self.data.claim.to_list(),
                self.data.text.to_list(),
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



nei_df = train_df[train_df['label']=='NOT ENOUGH INFO'].sample(frac=0.5)
ref_df = train_df[train_df['label']=='refutes'].sample(frac=0.5)
train_df = pd.concat([nei_df, ref_df, train_df]).sample(frac=1)
train_df = pd.concat([train_df]*1).sample(frac=1)


nei_df = dev_df[dev_df['label']=='NOT ENOUGH INFO'].sample(frac=0.5)
ref_df = dev_df[dev_df['label']=='refutes'].sample(frac=0.5)
dev_df = pd.concat([dev_df]*1).sample(frac=1)

train_df['target'] = train_df['label'].apply(lambda x: LABEL2ID[x])
dev_df['target'] = dev_df['label'].apply(lambda x: LABEL2ID[x])

train_df['text'] = train_df['evidence_list'].apply(lambda l: '\n'.join(l))
dev_df['text'] = dev_df['evidence_list'].apply(lambda l: '\n'.join(l))

text = train_df['claim']
claim = train_df['text']
target = train_df['target']
reverse_df = pd.DataFrame(data={'claim':claim.tolist(), 'text':text.tolist(), 'target':target.tolist()})
train_df = pd.concat([train_df, reverse_df]).sample(frac=1)

train_df['source1'] = train_df['claim'].apply(lambda x: CONVERTER_T2S.convert(x))
train_df['source2'] = train_df['text'].apply(lambda x: CONVERTER_T2S.convert(x))
dev_df['source1'] = dev_df['claim'].apply(lambda x: CONVERTER_T2S.convert(x))
dev_df['source2'] = dev_df['text'].apply(lambda x: CONVERTER_T2S.convert(x))

train_df = train_df.sample(frac=1)

# %%
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
model_path = '/nfs/nas-6.1/wclu/AICUP/2023/lert/ckeckpoints/lert/checkpoint-12272'
model = AutoModelForSequenceClassification.from_pretrained(model_path, cache_dir = '/nfs/nas-6.1/wclu/cache', num_labels=3)
tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir = '/nfs/nas-6.1/wclu/cache')

# %%
train_dataset = CustomDataset(train_df, tokenizer, 512, 512)
dev_dataset = CustomDataset(dev_df, tokenizer, 512, 512)

# %%
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = [x.argmax() for x in predictions]
    return {'val_acc': accuracy_score(predictions, labels)}

# %%
name = 'bs32-reversed-cosine'

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# %%
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir = f'checkpoints/{name}', 
    gradient_accumulation_steps = 2,
    evaluation_strategy = 'steps',
    eval_steps = 100,
    per_device_train_batch_size = 16,       
    per_device_eval_batch_size = 128,   
    learning_rate = 2e-5,
    weight_decay = 0.01,                                 
    num_train_epochs = 2.5,                           
    warmup_ratio = 0.1,  
    lr_scheduler_type = "cosine",                      
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


