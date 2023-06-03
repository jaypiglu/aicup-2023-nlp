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
model_name = 'lert-base-doc12-xnli_e2_500'

LABEL2ID: Dict[str, int] = {
    "supports": 0,
    "NOT ENOUGH INFO": 1,
    "refutes": 2
}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL2ID.items()}


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
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# %%
data_list = [
    'lert-large-doc12-xnli_e1_700',
    'lert-base-doc12-xnli_e2_500',
    'lert-base-doc12-xnli_e2_bs64_reversed'
]

model_list = [
    '/nfs/nas-6.1/wclu/AICUP/2023/NCKU-AICUP2023-baseline/ver/checkpoints/bs16-reversed-final/checkpoint-2600',
    '/nfs/nas-6.1/wclu/AICUP/2023/NCKU-AICUP2023-baseline/ver/checkpoints/bs32-reversed-final/checkpoint-1700',
    '/nfs/nas-6.1/wclu/AICUP/2023/NCKU-AICUP2023-baseline/ver/checkpoints/bs64-reversed-final/checkpoint-1300',
    
    '/nfs/nas-6.1/wclu/AICUP/2023/NCKU-AICUP2023-baseline/ver/checkpoints/bs16-reversed-init/checkpoint-3500',
    '/nfs/nas-6.1/wclu/AICUP/2023/NCKU-AICUP2023-baseline/ver/checkpoints/bs32-reversed-init/checkpoint-1600',
    '/nfs/nas-6.1/wclu/AICUP/2023/NCKU-AICUP2023-baseline/ver/checkpoints/bs64-reversed-init/checkpoint-1500',

    '/nfs/nas-6.1/wclu/AICUP/2023/NCKU-AICUP2023-baseline/ver/checkpoints/bs16-reversed-middle/checkpoint-4100',
    '/nfs/nas-6.1/wclu/AICUP/2023/NCKU-AICUP2023-baseline/ver/checkpoints/bs32-reversed-middle/checkpoint-3500',
    '/nfs/nas-6.1/wclu/AICUP/2023/NCKU-AICUP2023-baseline/ver/checkpoints/bs64-reversed-middle/checkpoint-1700',

    '/nfs/nas-6.1/wclu/AICUP/2023/NCKU-AICUP2023-baseline/ver/checkpoints/bs32-reversed-cosine/checkpoint-1700',
    '/nfs/nas-6.1/wclu/AICUP/2023/NCKU-AICUP2023-baseline/ver/checkpoints/bs64-reversed-cosine/checkpoint-1200'

]

# %%
from transformers import Trainer, TrainingArguments
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter
prob = []
mode = 'private'
for model_path in tqdm(model_list):
    model = AutoModelForSequenceClassification.from_pretrained(model_path, cache_dir = '/nfs/nas-6.1/wclu/cache', num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir = '/nfs/nas-6.1/wclu/cache')
    training_args = TrainingArguments(
        output_dir = f'checkpoints/inference', 
        per_device_eval_batch_size = 512,   
    )
    trainer = Trainer(
        model = model,                                       
        tokenizer = tokenizer,
        args = training_args,                                          
    )
    for data_path in data_list:
        TEST_DATA = load_json(f"../data/{mode}_test_doc{NUM_DOC}sent5_{data_path}.jsonl")
        TEST_PKL_FILE = Path(f"../data/{mode}_test_doc{NUM_DOC}sent5_{data_path}.pkl")
        if not TEST_PKL_FILE.exists():
            test_df = join_with_topk_evidence(
                pd.DataFrame(TEST_DATA),
                mapping,
                mode="eval",
                topk=EVIDENCE_TOPK,
            )
            test_df.to_pickle(TEST_PKL_FILE, protocol=4)
        else:
            with open(TEST_PKL_FILE, "rb") as f:
                test_df = pickle.load(f)
        test_df['target'] = [0 for j in range(len(test_df))]
        test_df['text'] = test_df['evidence_list'].apply(lambda l: '\n'.join(l))
        test_df['source1'] = test_df['claim'].apply(lambda x: CONVERTER_T2S.convert(x))
        test_df['source2'] = test_df['text'].apply(lambda x: CONVERTER_T2S.convert(x))
        test_dataset = CustomDataset(test_df, tokenizer, 512, 512)
        result = trainer.predict(test_dataset)
        prob.append(result.predictions)     
        predicted_label = [x.argmax() for x in result.predictions]
        print(Counter(predicted_label))   

# %%
import numpy as np
def softmax(vec):
    return np.exp(vec)/np.sum(np.exp(vec))
label = []
for i in tqdm(range(prob[0].shape[0])):
    label_prob = np.array([0.0, 0.0, 0.0])
    for j in range(len(prob)):
        label_prob += softmax(prob[j][i])
        #label_prob += prob[j][i]
    label.append(label_prob.argmax())
from collections import Counter


# %%
vote = []
for i in tqdm(range(prob[0].shape[0])):
    pre_label = []
    for j in range(len(prob)):
        pre_label.append(prob[j][i].argmax())
    vote.append(pre_label)


# %%
vote_o = [Counter(v).most_common()[0][0] for v in vote]

# %%

vote_new = []
thres_2 = 13
thres_1 = 14
for v in vote:
    if Counter(v)[2] > thres_2: 
        vote_new.append(2)
    elif Counter(v)[1] > thres_1:
        vote_new.append(1)
    else:
        vote_new.append(Counter(v).most_common()[0][0])

# %%
predicted_label = vote_new
data_path = 'lert-large-doc12-xnli_e1_700'
name = f'33vote_{thres_1}_{thres_2}_{data_path}'
TEST_DATA = load_json(f"../data/{mode}_test_doc12sent5_{data_path}.jsonl")
TEST_PKL_FILE = Path(f"../data/{mode}_test_doc12sent5_{data_path}.pkl")
if not TEST_PKL_FILE.exists():
    test_df = join_with_topk_evidence(
        pd.DataFrame(TEST_DATA),
        mapping,
        mode="eval",
        topk=EVIDENCE_TOPK,
    )
    test_df.to_pickle(TEST_PKL_FILE, protocol=4)
else:
    with open(TEST_PKL_FILE, "rb") as f:
        test_df = pickle.load(f)
OUTPUT_FILENAME = f'{mode}_submission_{name}.jsonl'
predict_dataset = test_df.copy()
predict_dataset["predicted_label"] = list(map(ID2LABEL.get, predicted_label))
predict_dataset[["id", "predicted_label", "predicted_evidence"]].to_json(
    OUTPUT_FILENAME,
    orient="records",
    lines=True,
    force_ascii=False,
)


predicted_label = vote_o
data_path = 'lert-large-doc12-xnli_e1_700'
name = f'33vote_original_{data_path}'
TEST_DATA = load_json(f"../data/{mode}_test_doc12sent5_{data_path}.jsonl")
TEST_PKL_FILE = Path(f"../data/{mode}_test_doc12sent5_{data_path}.pkl")
if not TEST_PKL_FILE.exists():
    test_df = join_with_topk_evidence(
        pd.DataFrame(TEST_DATA),
        mapping,
        mode="eval",
        topk=EVIDENCE_TOPK,
    )
    test_df.to_pickle(TEST_PKL_FILE, protocol=4)
else:
    with open(TEST_PKL_FILE, "rb") as f:
        test_df = pickle.load(f)
OUTPUT_FILENAME = f'{mode}_submission_{name}.jsonl'
predict_dataset = test_df.copy()
predict_dataset["predicted_label"] = list(map(ID2LABEL.get, predicted_label))
predict_dataset[["id", "predicted_label", "predicted_evidence"]].to_json(
    OUTPUT_FILENAME,
    orient="records",
    lines=True,
    force_ascii=False,
)

import pickle
with open('private_vote.pkl', 'wb') as f:
    pickle.dump(vote, f)
