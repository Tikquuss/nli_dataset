import os
import pandas as pd
import json
from tqdm import tqdm
import re
import spacy
from transformers import RobertaTokenizer

nlp = spacy.load("en_core_web_sm")

label_set = {"neutral", "contradiction", "entailment"}
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def check_tokens(text):
    doc = nlp(text)
    for token in doc:
        sub_tokens = tokenizer(token.text, add_special_tokens=False, return_tensors="pt")["input_ids"]
        if sub_tokens.shape[-1] == 0:
            return False
    return True


def parse_mnli(input_filepath:str):
    """Extract the sentence pair and the corresponding label
    Args:
        input_filepath: path of the file containing bulk data
        output_filepath: path of the file where the parsed data will be saved
    """
    def trim(text):
        out = text.strip()
        out = re.sub(" +", " ", out)
        return out
    
    paths = input_filepath.split("/")
    filename = ".".join(paths[-1].split(".")[:-1])
    paths[-1] = f"{filename}.csv"
    output_filepath = "/".join(paths)
    data = []
    for line in tqdm(open(input_filepath).readlines()):
        item = json.loads(line)
        if item["gold_label"] not in label_set:
            continue
        sentence1, sentence2 = trim(item["sentence1"]), trim(item["sentence2"])
        if check_tokens(sentence1) and check_tokens(sentence2):
            data.append([item["pairID"], sentence1, sentence2, item["gold_label"]])
    data = pd.DataFrame(data=data, columns=["pairID", "sentence1", "sentence2", "label"])
    data = data.dropna()
    data.to_csv(output_filepath, index=False)



parse_mnli("multinli_1.0/multinli_1.0_train.jsonl")
parse_mnli("multinli_1.0/multinli_1.0_dev_matched.jsonl")
parse_mnli("multinli_1.0/multinli_1.0_dev_mismatched.jsonl")


parse_mnli("snli_1.0/snli_1.0_dev.jsonl")
parse_mnli("snli_1.0/snli_1.0_test.jsonl")
parse_mnli("snli_1.0/snli_1.0_train.jsonl")