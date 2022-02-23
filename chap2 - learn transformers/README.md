## Chap 2
### Pipeline
A pipeline consists of
1. Tokenizer
2. Model
3. Pre-processing

### Tokenizer
- Split Text
- Create Vocabulary and mapping between text to int and int to text
- AutoTokenizer class and its from_pretrained() method. Using the checkpoint name of our model, it will automatically fetch the data associated with the model's tokenizer and cache it (so it's only downloaded the first time you run the code below).
- Encode text to ints or Decode int to text
- Some may add " [CLS] at the beginning and the special word [SEP] at the end.  " to begining and end

INPUT : list of text
OUTPUT : {
    input_ids : Tensor, #token to int
    attention_mask : Tensor #boolean
    maybe more
}

```
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```



### Model
model.config : Metadata about model and mapping from head ints to values

INPUT : Tensor of (batch, sequence length, hidden size)
OUTPUT : Tensor of logits


Model Types
*Model (retrieve the hidden states)
*ForCausalLM
*ForMaskedLM
*ForMultipleChoice
*ForQuestionAnswering
*ForSequenceClassification
*ForTokenClassification
and others 🤗


Loadinga  pre-trained model
```
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")
```

Saving
```
model.save_pretrained("directory_on_my_computer")
#outputs config.json tf_model.h5
```

### Pre-Processing
