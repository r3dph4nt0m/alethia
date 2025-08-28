import pandas as pd
import os
from transformers import GPT2Tokenizer  # Fixed import
from sklearn.model_selection import train_test_split
import torch
import requests
import ssl
import proxy_bypass

class PrepareDataset:
    def __init__(self, files, batch_size=75, train_size=0.8, test_size=0.1, val_size=0.1, max_length=512):
        self.files = files
        self.batch_size = batch_size
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        self.max_length = max_length
        
        # Configure requests to bypass proxy for huggingface.co
        proxy_bypass._configure_proxy_bypass()
        
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        #Load organized and cleaned language data
        khasi_data = pd.read_csv(files['khasi'], encoding='utf-8')
        welsh_data = pd.read_csv(files['welsh'], encoding='utf-8')
        ainu_data = pd.read_csv(files['ainu'], encoding='utf-8')
        self.languages_data = pd.concat([khasi_data, welsh_data, ainu_data], ignore_index=True)
        self.languages_data = self.languages_data.dropna().reset_index(drop=True)
        print(f"Total samples after loading and cleaning: {len(self.languages_data)}")

        self.dataset = { #create emptyy dataset
            "input_ids": torch.empty((0, self.max_length), dtype=torch.long),
            "attention_mask": torch.empty((0, self.max_length), dtype=torch.long),
            "classification_label": torch.empty((0, self.max_length), dtype=torch.long),
            "translation_label": torch.empty((0, self.max_length), dtype=torch.long),
        }
        self.save_file = "data/tokenized_dataset.pt"
        if not os.path.exists(self.save_file):
            torch.save(self.dataset, self.save_file)

    #process data into tokenized tensors for fine-tuning
    def tokenize_data(self):
        for i in range(0, len(self.languages_data), self.batch_size):
            batch = self.languages_data.iloc[i:i + self.batch_size]
            dialogues = batch['dialogue'].tolist()
            classifications = batch['classification'].tolist()
            translations = batch['translation'].tolist()
            #concatenate all features and labels into one for next-token prediction
            feature_set = [f"Dialogue: {d} \nClassification: {c} \nTranslation: {t}" for d, c, t in zip(dialogues, classifications, translations)]
            tokenized_feature_set = self.tokenizer(
                feature_set,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            classification_labels = self.tokenizer(
                [f"Classification: {c}" for c in classifications],
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )['input_ids']
            translation_labels = self.tokenizer(
                [f"Translation: {t}" for t in translations],
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )['input_ids']

            # Concatenate and save into existing dataset

            self.dataset["input_ids"] = torch.cat((self.dataset["input_ids"], tokenized_feature_set['input_ids']), dim=0)
            self.dataset["attention_mask"] = torch.cat((self.dataset["attention_mask"], tokenized_feature_set['attention_mask']), dim=0)
            self.dataset["classification_label"] = torch.cat((self.dataset["classification_label"], classification_labels), dim=0)
            self.dataset["translation_label"] = torch.cat((self.dataset["translation_label"], translation_labels), dim=0)
            current_dataset = torch.load(self.save_file)
            self.dataset = {key: torch.cat((current_dataset[key], self.dataset[key]), dim=0) for key in self.dataset.keys()}
            torch.save(self.dataset, self.save_file)
            self.dataset = {
                "input_ids": torch.empty((0, self.max_length), dtype=torch.long),
                "attention_mask": torch.empty((0, self.max_length), dtype=torch.long),
                "classification_label": torch.empty((0, self.max_length), dtype=torch.long),
                "translation_label": torch.empty((0, self.max_length), dtype=torch.long),
            }
            print(f"\rBatch {i // self.batch_size + 1}/{(len(self.languages_data) + self.batch_size - 1) // self.batch_size} tokenized and saved.", end='', flush=True)
        print(f"\nTokenization complete.")

if __name__ == "__main__":
    files = {
        "khasi": "data/structured_khasi.csv",
        "welsh": "data/structured_welsh.csv",
        "ainu": "data/structured_ainu.csv"
    }
    dataset_preparer = PrepareDataset(files)
    dataset_preparer.tokenize_data()
    tokenized_dataset = torch.load(dataset_preparer.save_file)
    print(f"Total tokenized samples: {len(tokenized_dataset['input_ids'])}")
    print(f"Input IDs shape: {tokenized_dataset['input_ids'].shape}")
