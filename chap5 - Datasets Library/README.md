## Loading Local Data

|Data format|Loading script|Example|
|-----------|--------------|-------|
|CSV & TSV|csv|load_dataset("csv", data_files="my_file.csv")|
|Text files|text|load_dataset("text", data_files="my_file.txt")|
|JSON & JSON Lines|json|load_dataset("json", data_files="my_file.jsonl")|
|Pickled DataFrames|pandas|load_dataset("pandas", data_files="my_dataframe.pkl")|

<br/>
<br/>

## Use Batched=True where possible - Rust!
|Options	|Fast tokenizer|Slow tokenizer|
|-----------|--------------|--------------|
|batched=True|	10.8s	|4min41s|
|batched=False|	59.2s|	5min3s|

<br/>
<br/>

## Use Multi-processing when cant use Rust
|Options	|Fast tokenizer|Slow tokenizer|
|-----------|--------------|--------------|
|batched=True	|10.8s	|4min41s|
|batched=False	|59.2s	|5min3s|
|batched=True, num_proc=8	|6.52s	|41.3s|
|batched=False, num_proc=8	|9.49s	|45.2s|


```
### Process with Pandas
drug_dataset.set_format("pandas")
```
Under the hood, Dataset.set_format() changes the return format for the dataset's `__getitem__()` dunder method. This means that when we want to create a new object like train_df from a Dataset in the "pandas" format, we need to slice the whole dataset to obtain a pandas.DataFrame. You can verify for yourself that the type of drug_dataset["train"] is Dataset, irrespective of the output format.

<br/>
<br/>

## Save DataSEt Formats

|Dataformat	|Function|
|-----------|--------------|
|Arrow|	Dataset.save_to_disk()|
|CSV|	Dataset.to_csv()|
|JSON|	Dataset.to_json()|


## Arrow Datasets - used for big Data
Datasets treats each dataset as a memory-mapped file, which provides a mapping between RAM and filesystem storage that allows the library to access and operate on elements of the dataset without needing to fully load it into memory.
Memory-mapped files can also be shared across multiple processes, which enables methods like Dataset.map() to be parallelized without needing to move or copy the dataset. Under the hood, these capabilities are all realized by the Apache Arrow memory format and pyarrow library, which make the data loading and processing lightning fast. (For more details about Apache Arrow and comparisons to Pandas, check out Dejan Simics blog post.) To see this in action, lets run a little speed test by iterating over all the elements in the PubMed Abstracts dataset:


## Create your own dataset
https://huggingface.co/course/chapter5/5?fw=tf