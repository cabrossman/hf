## Token Classification
**Any task which attributes a label to each token in a sentence**
- NER
- POS tagging
- Chunking
Datasets contain labels for the three tasks we mentioned earlier: NER, POS, and chunking. A big difference from other datasets is that the input texts are not presented as sentences or documents, but lists of words (the last column is called tokens, but it contains words in the sense that these are pre-tokenized inputs that still need to go through the tokenizer for subword tokenization).

## Fine Tuning a masked language model (Predict masked word)
a few cases where you'll want to first fine-tune the language models on your data, before training a task-specific head. For example, if your dataset contains legal contracts or scientific articles, a vanilla Transformer model like BERT will typically treat the domain-specific words in your corpus as rare tokens, and the resulting performance may be less than satisfactory. By fine-tuning the language model on in-domain data you can boost the performance of many downstream tasks, which means you usually only have to do this step once!