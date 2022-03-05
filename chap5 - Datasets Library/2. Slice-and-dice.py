from datasets import load_dataset, load_from_disk
import html

data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
# \t is the tab character in Python
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

###  small random subset
### Intervactive
"""
drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
drug_sample[:3]

# Output
{'Unnamed: 0': [87571, 178045, 80482],
 'drugName': ['Naproxen', 'Duloxetine', 'Mobic'],
 'condition': ['Gout, Acute', 'ibromyalgia', 'Inflammatory Conditions'],
 'review': ['"like the previous person mention, I&#039;m a strong believer of aleve, it works faster for my gout than the prescription meds I take. No more going to the doctor for refills.....Aleve works!"',
  '"I have taken Cymbalta for about a year and a half for fibromyalgia pain. It is great\r\nas a pain reducer and an anti-depressant, however, the side effects outweighed \r\nany benefit I got from it. I had trouble with restlessness, being tired constantly,\r\ndizziness, dry mouth, numbness and tingling in my feet, and horrible sweating. I am\r\nbeing weaned off of it now. Went from 60 mg to 30mg and now to 15 mg. I will be\r\noff completely in about a week. The fibro pain is coming back, but I would rather deal with it than the side effects."',
  '"I have been taking Mobic for over a year with no side effects other than an elevated blood pressure.  I had severe knee and ankle pain which completely went away after taking Mobic.  I attempted to stop the medication however pain returned after a few days."'],
 'rating': [9.0, 3.0, 10.0],
 'date': ['September 2, 2015', 'November 7, 2011', 'June 5, 2013'],
 'usefulCount': [36, 13, 128]}

 Clean up
    The Unnamed: 0 column looks suspiciously like an anonymized ID for each patient.
    The condition column includes a mix of uppercase and lowercase labels.
    The reviews are of varying length and contain a mix of Python line separators (\r\n) as well as HTML character codes like &\#039;.
"""
### Cleanup Function - overwrite existing column by returning existing key
def lowercase_condition(example):
    return {"condition": example["condition"].lower()}

### Create New Column by just returning new key
def compute_review_length(example):
    return {"review_length": len(example["review"].split())}


### Rename Column
drug_dataset = drug_dataset.rename_column(original_column_name="Unnamed: 0", new_column_name="patient_id")
### Remove Nones in condition
drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)
drug_dataset = drug_dataset.map(lowercase_condition)
### Creatte new Columns
drug_dataset = drug_dataset.map(compute_review_length)
### Remove examples with reviews under 30 chars
drug_dataset = drug_dataset.filter(lambda x: x["review_length"] > 30)
### REmove html chars - this would be slow....
drug_dataset = drug_dataset.map(lambda x: {"review": html.unescape(x["review"])})
### This is same thing sped up with batched which takes in number of rows at once
new_drug_dataset = drug_dataset.map(lambda x: {"review": [html.unescape(o) for o in x["review"]]}, batched=True)

### Interactive
"""
Check out extreeme cases
drug_dataset["train"].sort("review_length")[:3]
OUTPUT:
{'patient_id': [103488, 23627, 20558],
 'drugName': ['Loestrin 21 1 / 20', 'Chlorzoxazone', 'Nucynta'],
 'condition': ['birth control', 'muscle spasm', 'pain'],
 'review': ['"Excellent."', '"useless"', '"ok"'],
 'rating': [10.0, 1.0, 6.0],
 'date': ['November 4, 2008', 'March 24, 2017', 'August 20, 2016'],
 'usefulCount': [5, 2, 10],
 'review_length': [1, 1, 1]}
"""


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

### uses rust!
def tokenize_function(examples):
    return tokenizer(examples["review"], truncation=True)


# doesnt use rust - use_fast = Falase
slow_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=False)


### This doesnt use Rust but it uses multi processing. 
def slow_tokenize_function(examples):
    return slow_tokenizer(examples["review"], truncation=True)


tokenized_dataset = drug_dataset.map(slow_tokenize_function, batched=True, num_proc=8)


#
## Change dataset type
drug_dataset.set_format("pandas")
## Get pandas df
train_df = drug_dataset["train"][:]
## Change back to dataset
train_df = Dataset.from_pandas(train_df)
## Reset back to arrow
drug_dataset.reset_format()




#
## Creaate dataset splits
drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
# Rename the default "test" split to "validation"
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
# Add the "test" set to our `DatasetDict`
drug_dataset_clean["test"] = drug_dataset["test"]



#
## Save data
drug_dataset_clean.save_to_disk("drug-reviews")
drug_dataset_reloaded = load_from_disk("drug-reviews")

"""
drug-reviews/
├── dataset_dict.json
├── test
│   ├── dataset.arrow
│   ├── dataset_info.json
│   └── state.json
├── train
│   ├── dataset.arrow
│   ├── dataset_info.json
│   ├── indices.arrow
│   └── state.json
└── validation
    ├── dataset.arrow
    ├── dataset_info.json
    ├── indices.arrow
    └── state.json
"""