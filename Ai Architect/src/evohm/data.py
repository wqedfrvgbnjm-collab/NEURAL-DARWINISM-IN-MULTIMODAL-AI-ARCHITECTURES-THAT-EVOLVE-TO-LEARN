import json
import os
import random
from typing import Dict, Tuple

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import AutoTokenizer, CLIPProcessor


class HatefulMemesDataset(Dataset):
    def __init__(self, jsonl_path: str, img_dir: str, text_tokenizer, image_processor, max_seq_len: int, balance: bool = False):
        self.text_tokenizer = text_tokenizer
        self.image_processor = image_processor
        self.max_seq_len = max_seq_len
        self.img_dir = img_dir
        try:
            with open(jsonl_path) as f:
                self.data = [json.loads(line) for line in f]
        except FileNotFoundError:
            self.data = []

        valid_data = []
        for item in self.data:
            if 'img' in item and 'text' in item and 'label' in item:
                img_path = os.path.join(self.img_dir, item['img'])
                if os.path.exists(img_path):
                    valid_data.append(item)
        self.data = valid_data
        initial_size = len(self.data)
        print(f"Dataset: Found {initial_size} valid samples in {jsonl_path}.")

        if balance and initial_size > 0:
            print("--> Balancing requested. Applying oversampling...")
            labels = [d['label'] for d in self.data]
            class_counts = pd.Series(labels).value_counts()
            if len(class_counts) > 1:
                minority_class_label = class_counts.idxmin()
                majority_class_count = class_counts.max()
                minority_samples = [d for d in self.data if d['label'] == minority_class_label]
                num_to_oversample = majority_class_count - len(minority_samples)
                if num_to_oversample > 0:
                    oversampled_minority_samples = random.choices(minority_samples, k=num_to_oversample)
                    self.data.extend(oversampled_minority_samples)
                    random.shuffle(self.data)
                print(f"--> Dataset balanced. Original size: {initial_size}, New size: {len(self.data)}.")
            else:
                print("--> Only one class found. No balancing needed.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[Dict, Dict, Dict]:
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, item['img'])
        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, OSError):
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
        text_encoding = self.text_tokenizer(
            item['text'], return_tensors='pt', truncation=True, padding='max_length', max_length=self.max_seq_len
        )
        inputs = {
            'image': self.image_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0),
            'text_input_ids': text_encoding['input_ids'].squeeze(0),
            'text_attention_mask': text_encoding['attention_mask'].squeeze(0)
        }
        targets = {'classification': torch.tensor(item['label'], dtype=torch.long)}
        return inputs, targets, item


def _get_balanced_sampler(dataset):
    labels = [dataset[i][1]['classification'].item() for i in range(len(dataset))]
    class_counts = torch.bincount(torch.tensor(labels))
    class_weights = 1.0 / class_counts.float()
    sample_weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def get_data_loaders(data_root: str, train_file: str, val_file: str, image_dir: str, batch_size: int, seq_len: int, text_model_name: str, image_model_name: str, use_oversampling_for_train: bool = True, num_workers: int = 4):
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    image_processor = CLIPProcessor.from_pretrained(image_model_name)

    train_path = os.path.join(data_root, train_file)
    val_path = os.path.join(data_root, val_file)
    img_dir = os.path.join(data_root, image_dir)

    train_ds = HatefulMemesDataset(train_path, img_dir, text_tokenizer, image_processor, seq_len, balance=use_oversampling_for_train)
    valid_ds = HatefulMemesDataset(val_path, img_dir, text_tokenizer, image_processor, seq_len, balance=False)

    if len(train_ds) > 0:
        if use_oversampling_for_train:
            print("Using a standard shuffled DataLoader for the balanced training set.")
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        else:
            print("Using WeightedRandomSampler for the imbalanced training set.")
            train_sampler = _get_balanced_sampler(train_ds)
            train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True, drop_last=True)
    else:
        train_loader = None

    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False) if len(valid_ds) > 0 else None

    return train_loader, valid_loader
