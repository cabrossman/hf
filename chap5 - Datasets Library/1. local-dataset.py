from datasets import load_dataset

squad_it_dataset = load_dataset("json", data_files="SQuAD_it-train.json", field="data")

# Interactive
"""
# View Dataset
squad_it_dataset
#Ouput
DatasetDict({
    train: Dataset({
        features: ['title', 'paragraphs'],
        num_rows: 442
    })
})
#View Example
squad_it_dataset["train"][0]
#Output too big
"""

"""
Great, we've loaded our first local dataset! 
But while this worked for the training set, what we really want is to include both the train 
and test splits in a single DatasetDict object so we can apply Dataset.map() functions across 
both splits at once. To do this, we can provide a dictionary to the data_files argument 
that maps each split name to a file associated with that split:
"""
data_files = {"train": "SQuAD_it-train.json.gz", "test": "SQuAD_it-test.json.gz"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")

###Interactive
"""
squad_it_dataset
DatasetDict({
    train: Dataset({
        features: ['title', 'paragraphs'],
        num_rows: 442
    })
    test: Dataset({
        features: ['title', 'paragraphs'],
        num_rows: 48
    })
})
"""

# can load remote!
url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")