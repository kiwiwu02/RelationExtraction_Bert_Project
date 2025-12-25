from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import torch


class RelationshipDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, text_build_mode="basic"):
        self.sentences = [sample["sentence"] for sample in data]
        self.h_entities = [sample["h"] for sample in data]
        self.t_entities = [sample["t"] for sample in data]
        self.labels = [sample["r"] for sample in data]
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_map = {
            "临床表现": 0,
            "药物治疗": 1,
            "同义词": 2,
            "病因": 3,
            "并发症": 4,
            "病理分型": 5,
            "实验室检查": 6,
            "辅助治疗": 7,
            "相关（导致）": 8,
            "影像学检查": 9,
        }

        # Add special tags to tokenizer
        if text_build_mode == "entity_marked1":
            # First, check if these markers already exist
            new_tokens = ["[E1]", "[/E1]", "[E2]", "[/E2]"]
            tokens_to_add = []
            for token in new_tokens:
                if token not in self.tokenizer.get_vocab():
                    tokens_to_add.append(token)

            if tokens_to_add:
                self.tokenizer.add_special_tokens(
                    {"additional_special_tokens": tokens_to_add}
                )
        elif text_build_mode == "entity_marked2":
            new_tokens = ["[实体1]", "[/实体1]", "[实体2]", "[/实体2]"]
            tokens_to_add = []
            for token in new_tokens:
                if token not in self.tokenizer.get_vocab():
                    tokens_to_add.append(token)

            if tokens_to_add:
                self.tokenizer.add_special_tokens(
                    {"additional_special_tokens": tokens_to_add}
                )

        self.text_build_mode = text_build_mode

    def build_text(self, sentence, h, t):
        if self.text_build_mode == "basic1":
            # Basic Mode
            return f"[CLS] {h} [SEP] {sentence} [SEP] {t}"
        elif self.text_build_mode == "basic2":
            # Basic Mode2
            return f"[CLS] {h} [SEP] {t} [SEP] {sentence}"
        elif self.text_build_mode == "QA":
            # Q&A Template Mode
            return f"[CLS] {h}和{t}之间的关系是什么？[SEP] {sentence}"
        elif self.text_build_mode == "entity_marked1":
            # Entity tagging mode1
            marked_sentence = sentence.replace(h, f"[E1]{h}[/E1]").replace(
                t, f"[E2]{t}[/E2]"
            )
            return f"[CLS] {marked_sentence} [SEP]"
        elif self.text_build_mode == "entity_marked2":
            # Entity tagging mode2
            marked_sentence = sentence.replace(h, f"[实体1]{h}[/实体1]").replace(
                t, f"[实体2]{t}[/实体2]"
            )
            return f"[CLS] {marked_sentence} [SEP]"

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = self.sentences[item]
        h = self.h_entities[item]
        t = self.t_entities[item]
        label = self.label_map[self.labels[item]]  # Map relationship labels to numbers

        # Building Text
        text = self.build_text(sentence, h, t)

        # Using BERT tokenizer to encode text
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].flatten()
        attention_mask = encoded["attention_mask"].flatten()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
        }


class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, labels, batch_size):
        self.labels = labels
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_classes = len(set(labels))
        self.samples_per_class = batch_size // self.num_classes

        # Index data by category
        self.label_to_indices = {}
        for i in range(self.num_classes):
            self.label_to_indices[i] = np.where(np.array(labels) == i)[0]

        # Calculate the number of complete batches that can be generated for each category
        self.n_batches = (
            min([len(indices) for indices in self.label_to_indices.values()])
            // self.samples_per_class
        )

    def __iter__(self):
        # Create index copies for each category and shuffle them
        used_indices = {}
        available_indices = {}
        for label in range(self.num_classes):
            available_indices[label] = self.label_to_indices[label].copy()
            np.random.shuffle(available_indices[label])
            used_indices[label] = 0

        # Stop after generating a specified number of batches
        for _ in range(self.n_batches):
            indices = []
            for class_id in range(self.num_classes):
                class_indices = available_indices[class_id]
                indices.extend(
                    class_indices[
                        used_indices[class_id] : used_indices[class_id]
                        + self.samples_per_class
                    ]
                )
                used_indices[class_id] += self.samples_per_class

            yield indices

    def __len__(self):
        return self.n_batches


class PiorityBatchSampler(Sampler):
    def __init__(self, dataset, labels, batch_size=16):
        self.labels = labels
        self.dataset = dataset
        self.batch_size = batch_size  # Fixed at 16
        self.num_classes = len(set(labels))

        # Number of samples per batch for special categories (categories 4 and 8)
        self.special_classes = [4, 8]  # Categories indexed as 4 and 8
        self.samples_per_special = 4  # 4 samples for each special category

        # Number of samples per batch for other categories
        self.regular_classes = [
            i for i in range(self.num_classes) if i not in self.special_classes
        ]
        self.samples_per_regular = 1  # 1 sample each for other categories

        # Index data by category
        self.label_to_indices = {}
        for i in range(self.num_classes):
            self.label_to_indices[i] = np.where(np.array(labels) == i)[0]

        special_batches = min(
            [
                len(self.label_to_indices[i]) // self.samples_per_special
                for i in self.special_classes
            ]
        )
        regular_batches = min(
            [
                len(self.label_to_indices[i]) // self.samples_per_regular
                for i in self.regular_classes
            ]
        )
        self.n_batches = min(special_batches, regular_batches)

    def __iter__(self):
        # Create index copies for each category and shuffle them
        available_indices = {}
        used_indices = {}

        for label in range(self.num_classes):
            available_indices[label] = self.label_to_indices[label].copy()
            np.random.shuffle(available_indices[label])
            used_indices[label] = 0

        # Generate a specified number of batches
        for _ in range(self.n_batches):
            indices = []

            # Add samples of special categories (4 for each of categories 4 and 8)
            for class_id in self.special_classes:
                class_indices = available_indices[class_id]
                indices.extend(
                    class_indices[
                        used_indices[class_id] : used_indices[class_id]
                        + self.samples_per_special
                    ]
                )
                used_indices[class_id] += self.samples_per_special

            # Add samples from other categories (1 per category)

            for class_id in self.regular_classes:
                class_indices = available_indices[class_id]
                indices.extend(
                    class_indices[
                        used_indices[class_id] : used_indices[class_id]
                        + self.samples_per_regular
                    ]
                )
                used_indices[class_id] += self.samples_per_regular

            # Disrupt the order of samples in the current batch
            np.random.shuffle(indices)
            yield indices

    def __len__(self):
        return self.n_batches
