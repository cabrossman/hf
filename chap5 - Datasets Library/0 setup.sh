wget https://github.com/crux82/squad-it/raw/master/SQuAD_it-train.json.gz
wget https://github.com/crux82/squad-it/raw/master/SQuAD_it-test.json.gz

gzip -dkv SQuAD_it-*.json.gz


wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip"
unzip drugsCom_raw.zip