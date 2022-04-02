## Token Classification
**Any task which attributes a label to each token in a sentence**
- NER
- POS tagging
- Chunking
Datasets contain labels for the three tasks we mentioned earlier: NER, POS, and chunking. A big difference from other datasets is that the input texts are not presented as sentences or documents, but lists of words (the last column is called tokens, but it contains words in the sense that these are pre-tokenized inputs that still need to go through the tokenizer for subword tokenization).

## Fine Tuning a masked language model (Predict masked word)
a few cases where you'll want to first fine-tune the language models on your data, before training a task-specific head. For example, if your dataset contains legal contracts or scientific articles, a vanilla Transformer model like BERT will typically treat the domain-specific words in your corpus as rare tokens, and the resulting performance may be less than satisfactory. By fine-tuning the language model on in-domain data you can boost the performance of many downstream tasks, which means you usually only have to do this step once!

## Translation
Translation is another sequence-to-sequence task, which means its a problem that can be formulated as going from one sequence to another. In that sense the problem is pretty close to summarization, and you could adapt what we will see here to other sequence-to-sequence problems such as:

Style transfer: Creating a model that translates texts written in a certain style to another (e.g., formal to casual or Shakespearean English to modern English)
Generative question answering: Creating a model that generates answers to questions, given a context

## Summerization
In this section we'll take a look at how Transformer models can be used to condense long documents into summaries, a task known as text summarization. This is one of the most challenging NLP tasks as it requires a range of abilities, such as understanding long passages and generating coherent text that captures the main topics in a document. However, when done well, text summarization is a powerful tool that can speed up various business processes by relieving the burden of domain experts to read long documents in detail.

For ROUGE, recall measures how much of the reference summary is captured by the generated one. If we are just comparing words, recall can be calculated according to the following formula: (Number of Overlapping Words)/(Total words in ref summary). Percision : (Number of Overlapping Words)/(Total words in generated summary)

## Training a causal language model from scratch
Train a completely new model from scratch. This is a good approach to take if you have a lot of data and it is very different from the pretraining data used for the available models. However, it also requires considerably more compute resources to pretrain a language model than just to fine-tune an existing one. Examples where it can make sense to train a new model include for datasets consisting of musical notes, molecular sequences such as DNA, or programming languages. The latter have recently gained traction thanks to tools such as TabNine and GitHub's Copilot, powered by OpenAI's Codex model, that can generate long sequences of code. This task of text generation is best addressed with auto-regressive or causal language models such as GPT-2.

We can use the DataCollatorForLanguageModeling collator, which is designed specifically for language modeling (as the name subtly suggests). Besides stacking and padding batches, it also takes care of creating the language model labels — in causal language modeling the inputs serve as labels too (just shifted by one element), and this data collator creates them on the fly during training so we dont need to duplicate the input_ids. Shifting the inputs and labels to align them happens inside the model, so the data collator just copies the inputs to create the labels.

## Question & Answering
Extractive question answering. This involves posing questions about a document and identifying the answers as spans of text in the document itself.

(0, 0) if the answer is not in the corresponding span of the context
(start_position, end_position) if the answer is in the corresponding span of the context, with start_position being the index of the token (in the input IDs) at the start of the answer and end_position being the index of the token (in the input IDs) where the answer ends