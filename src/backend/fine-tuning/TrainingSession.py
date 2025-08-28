import torch
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments
from torch.utils.data import Dataset, random_split
import proxy_bypass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanguageDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

class TrainingSession:
    def __init__(self, model="gpt2", output_dir="model/language_classifier", epochs=20, batch_size=2, learning_rate=2e-04, warmup_steps=100, train_size=0.8, eval_size=0.1):
        self.dataset = torch.load("data/tokenized_dataset.pt")
        self.language_dataset = LanguageDataset(
            input_ids=self.dataset['input_ids'],
            attention_mask=self.dataset['attention_mask'],
            labels=self.dataset['input_ids']
        )
        self.train_size = int(train_size * len(self.language_dataset))
        self.eval_size = int(eval_size * len(self.language_dataset))
        self.test_size = len(self.language_dataset) - self.train_size - self.eval_size
        self.train_dataset, self.eval_dataset, self.test_dataset = random_split(
            self.language_dataset, [self.train_size, self.eval_size, self.test_size]
        )
        self.training_args = TrainingArguments(
            no_cuda=True,
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=10,
            save_steps=500,
            save_total_limit=3,
            eval_strategy="steps",
            eval_steps=500,  
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",  
            greater_is_better=False,  
            report_to=None,  
        )

        proxy_bypass._configure_proxy_bypass()
        self.model = GPT2LMHeadModel.from_pretrained(model)
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
        )

if __name__ == "__main__":
    session = TrainingSession()
    logger.info("Starting training session...")
    session.trainer.train()
    session.model.save_pretrained("model/language_classifier")
    session.trainer.save_state()