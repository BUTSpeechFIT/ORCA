import json
import random

import torch
from torch.utils.data import Dataset


class DataItem(dict):
    def to(self, *args, **kwargs):
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                self[key] = value.to(*args, **kwargs)
            elif isinstance(value, dict):
                self[key] = DataItem(value).to(*args, **kwargs)
            elif isinstance(value, list):
                self[key] = [
                    v.to(*args, **kwargs) if isinstance(v, torch.Tensor) else v for v in value
                ]
            else:
                self[key] = value
        return self


class UnifiedAnnotationDataset(Dataset):
    def __init__(
        self,
        annotations_file,
        prompts=None,
        filter_func=None,
        skip_rationale=False,
        skip_question=False,
        add_transcript=False,
    ):
        """
        A dataset that loads annotations from a JSON file and formats them for training.
        :param annotations_file: Path to the JSON file containing annotations.
        :param prompts: List of prompts to randomply sample a prompt to be prepended to the data. If None, a default empty prompt is used.
        :param filter_func: A function that takes an item and returns True if the item should be included in the dataset, False otherwise.
        """
        self.annotations = json.load(open(annotations_file, "r", encoding="utf-8"))
        self.annotation_keys = list(self.annotations.keys())
        self.annotation_keys = [
            key
            for key in self.annotation_keys
            if any([r is not None for r in self.annotations[key]["ratings"]])
        ]
        self.keys_to_id = {key: i for i, key in enumerate(self.annotation_keys)}
        self.annotation_ids = ["_".join(key.split("_")[:-2]) for key in self.annotation_keys]
        self.model_ids = ["_".join(key.split("_")[-2:]) for key in self.annotation_keys]
        self.prompts = prompts if prompts is not None else [""]
        self.context_prefix = "|context: "
        self.question_prefix = "|question: "
        self.answer_prefix = "|correct answer: "
        self.candidate_prefix = "|candidate answer: "
        self.transcript_prefix = "|transcription: "
        self.skip_rationale = skip_rationale
        self.skip_question = skip_question
        self.add_transcript = add_transcript

        if filter_func is not None:
            print("Length before filtering:", len(self.annotation_keys))
            self.annotation_keys = [
                key for key in self.annotation_keys if filter_func(self[self.keys_to_id[key]])
            ]
            self.keys_to_id = {key: i for i, key in enumerate(self.annotation_keys)}
            self.annotation_ids = ["_".join(key.split("_")[:-2]) for key in self.annotation_keys]
            self.model_ids = ["_".join(key.split("_")[-2:]) for key in self.annotation_keys]
            print("Length after filtering:", len(self.annotation_keys))

    def __len__(self):
        return len(self.annotation_keys)

    def keys(self):
        return self.annotation_keys

    def __getitem__(self, idx):
        item = self.annotations[self.annotation_keys[idx]]
        prompt = random.choice(self.prompts)
        context = item["rationale"]
        question = item["question"]
        answer = item["reference"]
        candidate_answer = item["candidate"]
        if self.add_transcript and "transcription" in item:
            context = f"{self.transcript_prefix}{item['transcription']} {context}"
        ratings = item["ratings"]
        ratings = [
            float(rating - 1) / 4 for rating in ratings if rating is not None and 0 <= rating <= 5
        ]
        average_rating = sum(ratings) / len(ratings)
        rating_variance = sum((x - average_rating) ** 2 for x in ratings) / len(ratings)
        # format_free = (
        #     f'{prompt} '
        #     f'{self.context_prefix}{context} '
        #     f'{self.question_prefix}{question} '
        #     f'{self.answer_prefix}{answer} '
        #     f'{self.candidate_prefix}{candidate_answer}'
        # )
        format_free = f"{prompt}"
        if not self.skip_rationale:
            format_free += f" {self.context_prefix}{context}"
        if not self.skip_question:
            format_free += f" {self.question_prefix}{question}"
        format_free += f" {self.answer_prefix}{answer}"
        format_free += f" {self.candidate_prefix}{candidate_answer}"

        model_id = self.model_ids[idx]
        return {
            "text": format_free,
            "soft_labels": ratings,
            "average_rating": average_rating,
            "rating_variance": rating_variance,
            "metadata": item,
            "prompt": prompt,
            "idx": self.annotation_keys[idx],
            "annotation_id": self.annotation_ids[idx],
            "model_id": model_id,
        }

    def __iter__(self):
        for i in range(len(self.annotation_keys)):
            yield self[i]


class InferenceDataset(UnifiedAnnotationDataset):
    def __getitem__(self, idx):
        item = self.annotations[self.annotation_keys[idx]]
        prompt = random.choice(self.prompts)
        context = item["rationale"]
        question = item["question"]
        answer = item["reference"]
        candidate_answer = item["candidate"]
        ratings = [0.0, 1.0]
        average_rating = sum(ratings) / len(ratings)
        rating_variance = sum((x - average_rating) ** 2 for x in ratings) / len(ratings)
        format_free = (
            f"{prompt} "
            f"{self.context_prefix}{context} "
            f"{self.question_prefix}{question} "
            f"{self.answer_prefix}{answer} "
            f"{self.candidate_prefix}{candidate_answer}"
        )
        model_id = self.model_ids[idx]
        return {
            "text": format_free,
            "soft_labels": ratings,
            "average_rating": average_rating,
            "rating_variance": rating_variance,
            "metadata": item,
            "prompt": prompt,
            "idx": self.annotation_keys[idx],
            "annotation_id": self.annotation_ids[idx],
            "model_id": model_id,
        }


class AToBDataset(Dataset):
    def __init__(self, input_dataset, output_dataset):
        """
        A dataset that combines two datasets, where the text and labels of the first are used as input and the labels of the second as output.
        :param input_dataset: Dataset containing input data with text and soft labels.
        :param output_dataset: Dataset containing ratings to be predicted.
        """
        self.input_dataset = input_dataset
        self.output_dataset = output_dataset
        self.input_keys = list(input_dataset.keys())
        self.output_keys = list(output_dataset.keys())
        self.keys = [k for k in self.input_keys if k in self.output_keys]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        input_idx = self.input_dataset.keys_to_id[key]
        output_idx = self.output_dataset.keys_to_id[key]
        input_item = self.input_dataset[input_idx]
        output_item = self.output_dataset[output_idx]
        input_text = input_item["text"]
        input_ratings = input_item["soft_labels"]
        input_text = input_text + " ".join(
            [f" |rating {i}: {rating}" for i, rating in enumerate(input_ratings)]
        )
        return {
            "text": input_text,
            "soft_labels": output_item["soft_labels"],
            "average_rating": output_item["average_rating"],
            "rating_variance": output_item["rating_variance"],
            "metadata": {"input": input_item["metadata"], "output": output_item["metadata"]},
            "prompt": input_item["prompt"],
            "idx": self.keys[idx],
            "annotation_id": input_item["annotation_id"],
            "model_id": input_item["model_id"],
        }


class AAndBDataset(Dataset):
    def __init__(self, dataset_a, dataset_b):
        """
        A dataset that combines two datasets, where the labels of both are returned separately.
        :param dataset_a: First dataset containing input data with text and soft labels.
        :param dataset_b: Second dataset containing input data with text and soft labels.
        """
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.keys_a = list(dataset_a.keys())
        self.keys_b = list(dataset_b.keys())
        self.keys = [k for k in self.keys_a if k in self.keys_b]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        idx_a = self.dataset_a.keys_to_id[key]
        idx_b = self.dataset_b.keys_to_id[key]
        item_a = self.dataset_a[idx_a]
        item_b = self.dataset_b[idx_b]
        text = item_a["text"]
        # assert item_a['text'] == item_b['text']
        soft_labels_a = item_a["soft_labels"]
        soft_labels_b = item_b["soft_labels"]
        average_rating_a = item_a["average_rating"]
        average_rating_b = item_b["average_rating"]
        rating_variance_a = item_a["rating_variance"]
        rating_variance_b = item_b["rating_variance"]
        return {
            "text": text,
            "soft_labels": soft_labels_a,
            "average_rating": average_rating_a,
            "rating_variance": rating_variance_a,
            "soft_labels_a": soft_labels_a,
            "average_rating_a": average_rating_a,
            "rating_variance_a": rating_variance_a,
            "soft_labels_b": soft_labels_b,
            "average_rating_b": average_rating_b,
            "rating_variance_b": rating_variance_b,
            "metadata": {"a": item_a["metadata"], "b": item_b["metadata"]},
            "prompt": item_a["prompt"] + " " + item_b["prompt"],
            "idx": self.keys[idx],
            "annotation_id": item_a["annotation_id"],
            "model_id": item_a["model_id"] + "_" + item_b["model_id"],
        }


class ConcatenatedDataset(Dataset):
    def __init__(self, datasets, make_unique=False):
        """
        A dataset that concatenates multiple datasets together.
        :param datasets: List of datasets to concatenate.
        :param make_unique: If True, will remove overlapping annotation keys from previous datasets (ORDER MATTERS! Priority is given to the last dataset).
        """
        if make_unique:
            for i in range(len(datasets)):
                for j in range(i):
                    datasets[j].annotation_keys = [
                        key
                        for key in datasets[j].annotation_keys
                        if key not in datasets[i].annotation_keys
                    ]
        self.datasets = datasets
        self.lengths = [len(dataset) for dataset in datasets]
        self.total_length = sum(self.lengths)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        for i, length in enumerate(self.lengths):
            if idx < length:
                return self.datasets[i][idx]
            idx -= length
        raise IndexError("Index out of range")

    def __iter__(self):
        for dataset in self.datasets:
            for item in dataset:
                yield item


class Converter(Dataset):
    def __init__(self, dataset, convert_func):
        """
        A dataset that applies a conversion function to each item in the underlying dataset.
        :param dataset: The underlying dataset to convert.
        :param convert_func: A function that takes an item and returns a converted item.
        """
        self.dataset = dataset
        self.convert_func = convert_func

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return self.convert_func(item)


class DatasetWithSampling(Dataset):
    def __init__(self, dataloaders, sampling_weights=None):
        """
        A dataset that allows for weighted sampling of items.
        :param dataloaders: The underlying dataloaders to sample from.
        :param sampling_weights: Optional list of weights for each item in the dataset. If None, uniform sampling is used.
        """
        self.dataloaders = dataloaders
        self.sampling_weights = (
            sampling_weights if sampling_weights is not None else [1.0] * len(dataloaders)
        )
        assert len(self.sampling_weights) == len(
            dataloaders
        ), f"Sampling weights length {len(self.sampling_weights)} does not match dataloaders length {len(dataloaders)}."
        self.dataloader_iters = [iter(loader) for loader in dataloaders]

    def __len__(self):
        """
        Get the total number of items in the dataset.
        :return: The total number of items.
        """
        return sum(len(loader.dataset) for loader in self.dataloaders)

    def __getitem__(self, item):
        """
        Get an item from the dataset, sampling according to the specified weights.
        :param item: The index of the item to retrieve.
        :return: The sampled item.
        """
        idx = random.choices(range(len(self.dataloaders)), weights=self.sampling_weights)[0]
        try:
            return next(self.dataloader_iters[idx])
        except StopIteration:
            # If the iterator is exhausted, reset it and try again
            self.dataloader_iters[idx] = iter(self.dataloaders[idx])
            return next(self.dataloader_iters[idx])


class CollateFn:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        tokenizer = self.tokenizer
        all_text = [item["text"] for item in batch]
        all_metadata = [item["metadata"] for item in batch]
        all_prompts = [item["prompt"] for item in batch]
        all_soft_labels = [item["soft_labels"] for item in batch]
        all_average_ratings = [item["average_rating"] for item in batch]
        all_ratings_variance = [item["rating_variance"] for item in batch]
        all_annotation_ids = [item["annotation_id"] for item in batch]
        all_model_ids = [item["model_id"] for item in batch]
        all_idxs = [item["idx"] for item in batch]

        tokenized = tokenizer.batch_encode_plus(
            all_text,
            add_special_tokens=False,
            padding="longest",
            return_tensors="pt",
            padding_side="right",
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        input_lengths = attention_mask.sum(dim=1)

        # # Using this instead of batch_encode_plus to avoid padding side inconsistency issues
        # tokenized = [tokenizer.encode(text, add_special_tokens=False,) for text in all_text]
        # input_lengths = torch.tensor([len(tokens) for tokens in tokenized], dtype=torch.long)
        # max_length = input_lengths.max().item()
        # attention_mask = input_lengths[:, None] > torch.arange(max_length, device=input_lengths.device)[None, :]

        # input_ids = torch.nn.utils.rnn.pad_sequence(
        #     [torch.tensor(tokens, dtype=torch.long) for tokens in tokenized],
        #     batch_first=True,
        #     padding_value=tokenizer.pad_token_id
        # )

        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(labels, dtype=torch.float) for labels in all_soft_labels],
            batch_first=True,
            padding_value=-100.0,
        )
        average_labels = torch.tensor(all_average_ratings, dtype=torch.float)
        label_variance = torch.tensor(all_ratings_variance, dtype=torch.float)

        extra_kwargs = {}
        if "soft_labels_a" in batch[0]:
            all_soft_labels_a = [item["soft_labels_a"] for item in batch]
            all_average_ratings_a = [item["average_rating_a"] for item in batch]
            all_ratings_variance_a = [item["rating_variance_a"] for item in batch]
            labels_a = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(labels, dtype=torch.float) for labels in all_soft_labels_a],
                batch_first=True,
                padding_value=-100.0,
            )
            average_labels_a = torch.tensor(all_average_ratings_a, dtype=torch.float)
            label_variance_a = torch.tensor(all_ratings_variance_a, dtype=torch.float)
            extra_kwargs.update(
                {
                    "labels_a": labels_a,
                    "average_labels_a": average_labels_a,
                    "label_variance_a": label_variance_a,
                }
            )
        if "soft_labels_b" in batch[0]:
            all_soft_labels_b = [item["soft_labels_b"] for item in batch]
            all_average_ratings_b = [item["average_rating_b"] for item in batch]
            all_ratings_variance_b = [item["rating_variance_b"] for item in batch]
            labels_b = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(labels, dtype=torch.float) for labels in all_soft_labels_b],
                batch_first=True,
                padding_value=-100.0,
            )
            average_labels_b = torch.tensor(all_average_ratings_b, dtype=torch.float)
            label_variance_b = torch.tensor(all_ratings_variance_b, dtype=torch.float)
            extra_kwargs.update(
                {
                    "labels_b": labels_b,
                    "average_labels_b": average_labels_b,
                    "label_variance_b": label_variance_b,
                }
            )

        return DataItem(
            {
                "input_ids": input_ids,
                "input_len": input_lengths,
                "attention_mask": attention_mask,
                "labels": labels,
                "average_labels": average_labels,
                "label_variance": label_variance,
                "metadata": all_metadata,
                "prompt": all_prompts,
                "annotation_id": all_annotation_ids,
                "model_id": all_model_ids,
                "idx": all_idxs,
                **extra_kwargs,
            }
        )
