import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPTNeoForCausalLM
import logging

# Setting up logging
loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.CRITICAL)
    
train_model = False  # Whether to train the model or not
push_to_hf = False  # If saving to your own huggingface repository
train_val_split = 0.9  # Currently trains on 90% of the data, validates on 10%
# This is a repository on Huggingface, model is downloaded automatically
loading_directory = "iocust/horos_gpt_neo"
# You can also use the checkpoint-80000 folder path (unzipped) from the repo's releases
# for loading the tokenizer and the model instead of the repository.
tokenizer = GPT2Tokenizer.from_pretrained(loading_directory, bos_token='<|startoftext|>',
                                          eos_token='<|endoftext|>', pad_token='<|pad|>')
model = GPTNeoForCausalLM.from_pretrained(loading_directory).cuda()
model.resize_token_embeddings(len(tokenizer))

if train_model:
    descriptions = pd.read_csv('./input/horoscopes_all_clean.csv', delimiter="\n", header=None)[0]
    max_length = max([len(tokenizer.encode(description)) for description in descriptions])


    class HoroscopeDataset(Dataset):
        def __init__(self, txt_list, tokenizer, max_length):
            self.input_ids = []
            self.attn_masks = []
            self.labels = []
            for txt in txt_list:
                encodings_dict = tokenizer('<|startoftext|>' + txt + '<|endoftext|>', truncation=True,
                                           max_length=max_length, padding="max_length")
                self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
                self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return self.input_ids[idx], self.attn_masks[idx]


    dataset = HoroscopeDataset(descriptions, tokenizer, max_length=max_length)
    train_size = int(train_val_split * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    training_args = TrainingArguments(output_dir='./training_output/checkpoints/', num_train_epochs=5, logging_steps=1000, save_steps=20000,
                                      per_device_train_batch_size=1, per_device_eval_batch_size=1,
                                      warmup_steps=100, weight_decay=0.01, logging_dir='./logs')
    Trainer(model=model, args=training_args, train_dataset=train_dataset,
            eval_dataset=val_dataset, data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                                                  'attention_mask': torch.stack([f[1] for f in data]),
                                                                  'labels': torch.stack([f[0] for f in data])}).train()

if push_to_hf:
    model.push_to_hub("horos_gpt_neo")
    tokenizer.push_to_hub("horos_gpt_neo")
