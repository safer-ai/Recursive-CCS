#%%
from datasets import load_dataset
from promptsource.templates import DatasetTemplates

ds_name = "amazon_polarity"
subset = None
if "/" in ds_name:
    ds_name, subset = ds_name.split("/")
ds = load_dataset(ds_name, subset)
for k in ["train", "val", "validation", "test"]:
    if k in ds:
        print(k, len(ds[k]))
train_key = "train"
if "train" not in ds:
    train_key = "auxiliary_train"
label_key = "label"
if "label" not in ds[train_key].features:
    label_key = "answer"
n_labels = len(ds[train_key].features[label_key].names)
print("n labels", n_labels)
for i in range(n_labels):
    print(ds[train_key].features[label_key].names[i], len([x for x in ds[train_key][label_key] if x == i]))

print(ds[train_key][0])
templates = DatasetTemplates(ds_name, subset).templates

idx = 25
for i, template in enumerate(list(templates.values())):
    print("--")
    try:
        prompt, r = template.apply(ds[train_key][idx])
        print(prompt)
        print("-->", r)
    except:
        print(template.apply(ds[train_key][idx]))
    print("--", i)
    print()

print(ds[train_key].features[label_key].names)
# %%
# ag_news, 100k, 4 labels # topic-classification
# dbpedia_14, 560k, 14 labels # text-classification
# imdb, 25k, 2 labels # sentiment
# tweet_eval/sentiment, 45k, 3 labels # sentiment
# tweet_eval/emotion, 4k, 4 labels # sentiment
# super_glue/rte, 3k, 2 labels # entailment
