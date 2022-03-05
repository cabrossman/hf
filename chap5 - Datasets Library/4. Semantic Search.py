from huggingface_hub import hf_hub_url
from datasets import load_dataset, Dataset

data_files = hf_hub_url(
    repo_id="lewtun/github-issues",
    filename="datasets-issues-with-comments.jsonl",
    repo_type="dataset",
)


issues_dataset = load_dataset("json", data_files=data_files, split="train")
issues_dataset = issues_dataset.filter(
    lambda x: (x["is_pull_request"] == False and len(x["comments"]) > 0)
)
columns = issues_dataset.column_names
columns_to_keep = ["title", "body", "html_url", "comments"]
columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
issues_dataset = issues_dataset.remove_columns(columns_to_remove)

## Get pandas explode function
issues_dataset.set_format("pandas")
df = issues_dataset[:]
comments_df = df.explode("comments", ignore_index=True)
comments_dataset = Dataset.from_pandas(comments_df)
comments_dataset = comments_dataset.map(lambda x: {"comment_length": len(x["comments"].split())})
comments_dataset = comments_dataset.filter(lambda x: x["comment_length"] > 15)

def concatenate_text(examples):
    return {
        "text": examples["title"]
        + " \n "
        + examples["body"]
        + " \n "
        + examples["comments"]
    }
comments_dataset = comments_dataset.map(concatenate_text)

## Create Embedding!
from transformers import AutoTokenizer, TFAutoModel
model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = TFAutoModel.from_pretrained(model_ckpt, from_pt=True)

### Pool on last hidden state [CLS]
def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

#helper function - tokenize a list of documents, place the tensors on the GPU, feed them to the model, and finally apply CLS pooling to the outputs:
def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="tf"
    )
    encoded_input = {k: v for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)

### Get single example
embedding = get_embeddings(comments_dataset["text"][0])
embedding.shape

### Embed Corpus
embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).numpy()[0]}
)

### Convert to faiss and find nearest neighbors
embeddings_dataset.add_faiss_index(column="embeddings")
question = "How can I load a dataset offline?"
question_embedding = get_embeddings([question]).numpy()
scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings", question_embedding, k=5
)