# %%
import pandas as pd
import json
import re
from tqdm.auto import tqdm


lines = []
for i in range(1,10):
    with open(f"/nfs/nas-6.1/wclu/AICUP/2023/aicup-2023-nlp/data/wiki-pages/wiki-00{i}.jsonl", encoding="utf8") as f:
        lines.extend(f.read().splitlines())
for i in range(10,25):
    with open(f"/nfs/nas-6.1/wclu/AICUP/2023/aicup-2023-nlp/data/wiki-pages/wiki-0{i}.jsonl", encoding="utf8") as f:
        lines.extend(f.read().splitlines())

wiki_df_0 = pd.DataFrame([json.loads(line) for line in lines])


# %%
wiki_text = []
lines = wiki_df_0['lines'].to_list()
ids = wiki_df_0['id'].to_list()

for i in range(len(lines)):
    for j in re.split(r"\n(?=[0-9])", lines[i]):
        j_list = j.split('\t')
        if len(j_list)>=2 and j_list[1]!='':
            wiki_text.append([ids[i], j_list[0], j_list[1]])


# %%
wiki_df = pd.DataFrame(wiki_text, columns=['id', 'line', 'text'])


from ckip_transformers.nlp import CkipWordSegmenter

ws_driver  = CkipWordSegmenter(model="bert-base", device=0)

#corpus = wiki_df['text'].tolist()
corpus = wiki_df_0['text'].tolist()

from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True)

corpus_tokenized = ws_driver(corpus, batch_size=512, max_length=509)

pd.Series(corpus_tokenized).parallel_apply(lambda x: ' '.join(x)).to_csv('data/tokenized_doc.csv', index=False)



