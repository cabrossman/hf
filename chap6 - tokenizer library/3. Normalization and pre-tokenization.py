from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print(type(tokenizer.backend_tokenizer))

print(tokenizer.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?")) 
#-> hello how are u?
print(tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?"))
#-> [('Hello', (0, 5)), (',', (5, 6)), ('how', (7, 10)), ('are', (11, 14)), ('you', (16, 19)), ('?', (19, 20))]

### DIfferent tokenizer -> GPT2
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
#it will split on whitespace and punctuation as well, 
#but it will keep the spaces and replace them with a Ġ symbol, 
#enabling it to recover the original spaces if we decode the tokens:
#-> [('Hello', (0, 5)),(',', (5, 6)), ('Ġhow', (6, 10)), ('Ġare', (10, 14)), ('Ġ', (14, 15)),('Ġyou', (15, 19)), ('?', (19, 20))]

### Different tokenizer -> t5-small
tokenizer = AutoTokenizer.from_pretrained("t5-small")
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
#Like the GPT-2 tokenizer, this one keeps spaces and replaces them with a specific 
#token (_), but the T5 tokenizer only splits on whitespace, not punctuation. 
# Also note that it added a space by default at the beginning of the sentence (before Hello) 
# and ignored the double space between are and you.
#-> [('▁Hello,', (0, 6)),('▁how', (7, 10)), ('▁are', (11, 14)),('▁you?', (16, 20))]