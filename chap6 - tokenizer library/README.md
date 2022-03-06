## Training a new tokenizer from an old one
If a language model is not available in the language you are interested in, or if your corpus is very different from the one your language model was trained on, you will most likely want to retrain the model from scratch using a tokenizer adapted to your data. 

## Fast Tokenizers
The output of a tokenizer isn't a simple Python dictionary; what we get is actually a special BatchEncoding object. It's a subclass of a dictionary (which is why we were able to index into that result without any problem before), but with additional methods that are mostly used by fast tokenizers.

Besides their parallelization capabilities, the key functionality of fast tokenizers is that they always keep track of the original span of texts the final tokens come from — a feature we call offset mapping. This in turn unlocks features like mapping each word to the tokens it generated or mapping each character of the original text to the token it’s inside, and vice versa.

### FUNCTIONS
`encoding.tokens()`
`encoding.word_ids()`
```
start, end = encoding.word_to_chars(3)
example[start:end]
```

## ALGOS

### Stenence Piece
- replaces spaces with _
- no need for pre-tokenization step
- reversible tokenization -> just concate tokens and replace "_" with " " and you have normalized text


### BPE - Byte-Pair Encoding Teokenization
Byte-Pair Encoding (BPE) was initially developed as an algorithm to compress texts, and then used by OpenAI for tokenization when pretraining the GPT model. Its used by a lot of Transformer models, including GPT, GPT-2, RoBERTa, BART, and DeBERTa.

BPE training starts by computing the unique set of words used in the corpus (after the normalization and pre-tokenization steps are completed), then building the vocabulary by taking all the symbols used to write those words. 

After getting this base vocabulary, we add new tokens until the desired vocabulary size is reached by learning merges, which are rules to merge two elements of the existing vocabulary together into a new one. So, at the beginning these merges will create tokens with two characters, and then, as training progresses, longer subwords.

At any step during the tokenizer training, the BPE algorithm will search for the most frequent pair of existing tokens (by 'pair,' here we mean two consecutive tokens in a word). That most frequent pair is the one that will be merged, and we rinse and repeat for the next step.


### WordPiece
Like BPE, WordPiece starts from a small vocabulary including the special tokens used by the model and the initial alphabet. Since it identifies subwords by adding a prefix (like ## for BERT), each word is initially split by adding that prefix to all the characters inside the word. So, for instance, "word" gets split like this:

`w ##o ##r ##d`

Thus, the initial alphabet contains all the characters present at the beginning of a word and the characters present inside a word preceded by the WordPiece prefix.
`score=(freq_of_pair)/(freq_of_first_element×freq_of_second_element)`
By dividing the frequency of the pair by the product of the frequencies of each of its parts, the algorithm prioritizes the merging of pairs where the individual parts are less frequent in the vocabulary.

Tokenization differs in WordPiece and BPE in that WordPiece only saves the final vocabulary, not the merge rules learned. Starting from the word to tokenize, WordPiece finds the longest subword that is in the vocabulary, then splits on it. For instance, if we use the vocabulary learned in the example above, for the word "hugs" the longest subword starting from the beginning that is inside the vocabulary is "hug", so we split there and get ["hug", "##s"]. We then continue with "##s", which is in the vocabulary, so the tokenization of "hugs" is ["hug", "##s"].

Tokenization differs in WordPiece and BPE in that WordPiece only saves the final vocabulary, not the merge rules learned. Starting from the word to tokenize, WordPiece finds the longest subword that is in the vocabulary, then splits on it. For instance, if we use the vocabulary learned in the example above, for the word "hugs" the longest subword starting from the beginning that is inside the vocabulary is "hug", so we split there and get ["hug", "##s"]. We then continue with "##s", which is in the vocabulary, so the tokenization of "hugs" is ["hug", "##s"].

With BPE, we would have applied the merges learned in order and tokenized this as ["hu", "##gs"], so the encoding is different.

As another example, lets see how the word "bugs" would be tokenized. "b" is the longest subword starting at the beginning of the word that is in the vocabulary, so we split there and get ["b", "##ugs"]. Then "##u" is the longest subword starting at the beginning of "##ugs" that is in the vocabulary, so we split there and get ["b", "##u, "##gs"]. Finally, "##gs" is in the vocabulary, so this last list is the tokenization of "bugs".

When the tokenization gets to a stage where its not possible to find a subword in the vocabulary, the whole word is tokenized as unknown — so, for instance, "mug" would be tokenized as ["[UNK]"], as would "bum" (even if we can begin with "b" and "##u", "##m" is not the vocabulary, and the resulting tokenization will just be ["[UNK]"], not ["b", "##u", "[UNK]"]). This is another difference from BPE, which would only classify the individual characters not in the vocabulary as unknown.

## Unigram Tokenization
The Unigram algorithm is often used in SentencePiece, which is the tokenization algorithm used by models like AlBERT, T5, mBART, Big Bird, and XLNet.

Compared to BPE and WordPiece, Unigram works in the other direction: it starts from a big vocabulary and removes tokens from it until it reaches the desired vocabulary size. There are several options to use to build that base vocabulary: we can take the most common substrings in pre-tokenized words, for instance, or apply BPE on the initial corpus with a large vocabulary size.

At each step of the training, the Unigram algorithm computes a loss over the corpus given the current vocabulary. Then, for each symbol in the vocabulary, the algorithm computes how much the overall loss would increase if the symbol was removed, and looks for the symbols that would increase it the least. Those symbols have a lower effect on the overall loss over the corpus, so in a sense they are 'less needed' and are the best candidates for removal.

This is all a very costly operation, so we dont just remove the single symbol associated with the lowest loss increase, but the pp (pp being a hyperparameter you can control, usually 10 or 20) percent of the symbols associated with the lowest loss increase. This process is then repeated until the vocabulary has reached the desired size.

Now, to tokenize a given word, we look at all the possible segmentations into tokens and compute the probability of each according to the Unigram model. Since all tokens are considered independent, this probability is just the product of the probability of each token. For instance, the tokenization ["p", "u", "g"] of "pug" has the probability:
P('p')*P('u')*P('g') = (5/210)*(36/210)*(20/210) = .000389

There is a classic algorithm used for this, called the Viterbi algorithm. Essentially, we can build a graph to detect the possible segmentations of a given word by saying there is a branch from character a to character b if the subword from a to b is in the vocabulary, and attribute to that branch the probability of the subwordnh.


## Algo Summary
|Model	|BPE|	WordPiece|	Unigram|
|----|----|----|----|
|Training|	Starts from a small vocabulary and learns rules to merge tokens|	Starts from a small vocabulary and learns rules to merge tokens|	Starts from a large vocabulary and learns rules to remove tokens|
|Training step|	Merges the tokens corresponding to the most common pair|	Merges the tokens corresponding to the pair with the best score based on the frequency of the pair, privileging pairs where each individual token is less frequent|	Removes all the tokens in the vocabulary that will minimize the loss computed on the whole corpus|
|Learns|	Merge rules and a vocabulary|	Just a vocabulary|	A vocabulary with a score for each token|
|Encoding|	Splits a word into characters and applies the merges learned during training|	Finds the longest subword starting from the beginning that is in the vocabulary, then does the same for the rest of the word|	Finds the most likely split into tokens, using the scores learned during training|

## Whole Pipeine
More precisely, the library is built around a central Tokenizer class with the building blocks regrouped in submodules:

normalizers contains all the possible types of Normalizer you can use (complete list here).
pre_tokenizers contains all the possible types of PreTokenizer you can use (complete list here).
models contains the various types of Model you can use, like BPE, WordPiece, and Unigram (complete list here).
trainers contains all the different types of Trainer you can use to train your model on a corpus (one per type of model; complete list here).
post_processors contains the various types of PostProcessor you can use (complete list here).
decoders contains the various types of Decoder you can use to decode the outputs of tokenization (complete list here).
You can find the whole list of building blocks here.