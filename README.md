## Common NLP Tasks
- Classifying whole sentences: Getting the sentiment of a review, detecting if an email is spam, determining if a sentence is grammatically correct or whether two sentences are logically related or not
- Classifying each word in a sentence: Identifying the grammatical components of a sentence (noun, verb, adjective), or the named entities (person, location, organization)
- Generating text content: Completing a prompt with auto-generated text, filling in the blanks in a text with masked words
- Extracting an answer from a text: Given a question and a context, extracting the answer to the question based on the information provided in the context
- Generating a new sentence from an input text: Translating a text into another language, summarizing a text

### Pipelines
- It connects a model with its necessary preprocessing and postprocessing steps, allowing us to directly input any text and get an intelligible answer:
```
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")
#[{'label': 'POSITIVE', 'score': 0.9598047137260437}]
```
- 3 Steps
    - Preprocessing
    - Model Predicts
    - Model Post Processing
- Available Pipelines
    - feature-extraction (get the vector representation of a text)
    - fill-mask
    - ner (named entity recognition)
    - question-answering
    - sentiment-analysis
    - summarization
    - text-generation
    - translation
    - zero-shot-classification
### Example Pipeline
```
### Pre Processing Function
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

### Preprocess the dataset
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)
### Get Datasets for training
from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer, return_tensors="tf"
)
#### Use Data Collator
tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
    collate_fn=data_collator,
    shuffle=True,
    batch_size=16,
)

tf_eval_dataset = tokenized_datasets["validation"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels", "token_type_ids"],
    collate_fn=data_collator,
    shuffle=False,
    batch_size=16,
)

### Get Base Model
from transformers import TFAutoModelForTokenClassification

model = TFAutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label={str(i): label for i, label in enumerate(label_names)},
    label2id={v: k for k, v in id2label.items()},
)

### Train Model
from transformers import create_optimizer
import tensorflow as tf
from transformers.keras_callbacks import PushToHubCallback
tf.keras.mixed_precision.set_global_policy("mixed_float16")

num_epochs = 3
num_train_steps = len(tf_train_dataset) * num_epochs

optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=num_train_steps, weight_decay_rate=0.01,)
model.compile(optimizer=optimizer)

callback = PushToHubCallback(output_dir="bert-finetuned-ner", tokenizer=tokenizer)

model.fit(tf_train_dataset, validation_data=tf_eval_dataset, callbacks=[callback], epochs=num_epochs,)

```
## Transformers
- Trained Self Supervised Learning
- Fine-tuning, on the other hand, is the training done after a model has been pretrained.
- Encoder: The encoder receives an input and builds a representation of it (its features). This means that the model is optimized to acquire understanding from the input.
- Decoder: The decoder uses the encoder's representation (features) along with other inputs to generate a target sequence. This means that the model is optimized for generating outputs.
### Model Types
- Encoder-only models: Good for tasks that require understanding of the input, such as sentence classification and named entity recognition.
- Decoder-only models: Good for generative tasks such as text generation.
- Encoder-decoder models or sequence-to-sequence models: Good for generative tasks that require an input, such as translation or summarization.

|Model	|Examples	|Tasks|
|----|----|----|
|Encoder|	ALBERT, BERT, DistilBERT, ELECTRA, RoBERTa|	Sentence classification, named entity recognition, extractive question answering|
|Decoder	|CTRL, GPT, GPT-2, Transformer XL|	Text generation|
|Encoder-decoder	|BART, T5, Marian, mBART|	Summarization, translation, generative question answering|

## Tokenization
- Normalization : strip, lowercasing, unicode
```
tokenizer.normalizer = normalizers.Sequence(
    [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
)
```
- Pre-Tokenization : split into words - or may replace with "_dag" or other token in existing word : `('▁Hello,', (0, 6))`
```
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
```
- Model : split words into subword tokens. To Train a Model:
```
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
```
- Postprocessing : add CLS & SEP. Train like so
```
tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
)
```
- Lastly add a decoder
```
tokenizer.decoder = decoders.WordPiece(prefix="##")
```
- TEST : 
```
encoding = tokenizer.encode("Let's test this tokenizer.")
# ['[CLS]', 'let', "'", 's', 'test', 'this', 'tok', '##eni', '##zer', '.', '[SEP]']
```
- SAVE & LOAD:
```
tokenizer.save("tokenizer.json")
new_tokenizer = Tokenizer.from_file("tokenizer.json")
```

- Split String into subwords: frequently used words should not be split into smaller subwords, but rare words should be decomposed into meaningful subwords.
- Assign a monotonically increasing int as vocabulary for each token
- May optionally have `type` and `mask`
- use `AutoTokenizer.from_pretrained(checkpoint)` to get the same tokenizer
- use offsets to get start and end of entities in orginal sentence
- `model_inputs = tokenizer(sequences, padding=True, truncation=True, return_tensors="tf", return_offsets_mapping=True)`
- Can handle sentence pairs:`tokenizer(sentence1_list, sentence2_list, padding=True. truncation=True)`
- `overflow_to_sample_mapping` used to map multiple chunks to single document
### Training a tokenizer
- If a language model is not available in the language you are interested in, or if your corpus is very different from the one your language model was trained on, you will most likely want to retrain the model from scratch using a tokenizer adapted to your data. 

- The first thing we need to do is transform the dataset into an iterator of lists of texts
- `old_tokenizer.train_new_from_iterator(training_corpus, 52000)`

### Pre-processing
- Padding to ensure same length - will include `tokenizer.pad_token_id`
- Masking : 1s = pay attention, 0s = skip
- Type : can indicate which observation the original word is from
- max sequence length - to truncate
- Add `[CLS]` & `[SEP]`
- The function that is responsible for putting together samples inside a batch is called a collate function. We have deliberately postponed the padding, to only apply it as necessary on each batch and avoid having over-long inputs with a lot of padding.  
- Dynamic padding means the samples in this batch should all be padded to a length of 67, the maximum length inside the batch. Without dynamic padding, all of the samples would have to be padded to the maximum length in the whole dataset, or the maximum length the model can accept.
- 

### Algorithms

|Model	|BPE|	WordPiece|	Unigram|
|----|----|----|----|
|Training|	Starts from a small vocabulary and learns rules to merge tokens|	Starts from a small vocabulary and learns rules to merge tokens|	Starts from a large vocabulary and learns rules to remove tokens|
|Training step|	Merges the tokens corresponding to the most common pair|	Merges the tokens corresponding to the pair with the best score based on the frequency of the pair, privileging pairs where each individual token is less frequent|	Removes all the tokens in the vocabulary that will minimize the loss computed on the whole corpus|
|Learns|	Merge rules and a vocabulary|	Just a vocabulary|	A vocabulary with a score for each token|
|Encoding|	Splits a word into characters and applies the merges learned during training|	Finds the longest subword starting from the beginning that is in the vocabulary, then does the same for the rest of the word|	Finds the most likely split into tokens, using the scores learned during training|

## Model
- `model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint)`
```
from tensorflow.keras.losses import SparseCategoricalCrossentropy

model.compile(
    optimizer="adam",
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.fit(
    tf_train_dataset,
    validation_data=tf_validation_dataset,
)
```


## Datasets
- Can load from hub or remote (github) or local
- Use Batch where possible
- To create new column have function with new key in object returned. To overwrite existing - return output with same key
- `.shuffle(seed=42).select(range(1000)).rename_column(...).map(ftn).filter(ftn).sort(key)[:3]`
- Streaming for big data : `pile_dataset = load_dataset("json", data_files=data_files, split="train", streaming=True)`. Then iterate over the returned : `next(iter(pile_dataset["train"]))`

### Batching & DataCollator
```
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
import numpy as np

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["labels"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)

tf_validation_dataset = tokenized_datasets["validation"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "token_type_ids"],
    label_cols=["labels"],
    shuffle=False,
    collate_fn=data_collator,
    batch_size=8,
)

```

## Model Specific
### Text Classification (NER, POS, Chunking, Embedding)
- Lables is list ([3, 0, 7, 0, 0, 0, 7, 0, 0]) which correspond to labels : ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
- DataCollatorWithPadding & DataCollatorForTokenClassification : labels needed to be padded same as input, using -100 as value corresponding to predictions being ignored in Loss. 
### Question & Answer
- Model predicts start and end of words in context. Thus is returns two logits per prediction - one corresponding to start and another corresponding to the end
- Input : `[CLS] question [SEP] context [SEP]` mask the tokens of the question as well as the [SEP] token. We'll keep the [CLS] token, however, as some models use it to indicate that the answer is not in the context.
- Compute all options that have highest 
`log(start.logit)+log(end.logit) where start>end`
- Need to truncate context but not remove answer during truncation. To do so there is truncate into smaller chunks, specifying the maximum length and a sliding window. `tokenizer(sentence, truncation=True, return_overflowing_tokens=True, max_length=6, stride=2
)`. This adds an `overflow_to_sample_mapping` to the output.