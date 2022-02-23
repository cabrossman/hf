## Chap 3 - pretraining

### Dataset Library

```
from datasets import load_dataset
raw_datasets = load_dataset("glue", "mrpc")
# raw_datasets - returns representation 
# raw_datasets["train"][0] - returns actual example 
# raw_datasets["train"].features - returns featurs
```

This command downloads and caches the dataset, by default in ~/.cache/huggingface/dataset.

raw_train_dataset.features