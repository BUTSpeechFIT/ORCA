"""Author: Bolaji Yusuf"""

import json
import random

import torch
from torch.utils.data import Dataset, Sampler


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
        log_fn=None,
        normalize_ratings=True,
    ):
        """
        A dataset that loads annotations from a JSONL file and formats them for training.

        Args:
            annotations_file: Path to the JSONL file containing annotations.
            prompts: List of prompts to randomly sample a prompt to be prepended to the data. If None, a default empty prompt is used.
            filter_func: A function that takes an item and returns True if the item should be included in the dataset, False otherwise.
            log_fn: Optional logging function. If provided, will log filtering statistics.
            normalize_ratings: If True, will normalize ratings originally in the range [1, 5] to the range [0, 1] by subtracting 1 and dividing by 4. If False, will keep original rating values.
                NOTE: pass normalize_ratings=False when using score_type='multinomial' so that
                raw integer ratings [1,5] are preserved; multinomial_llh in model.py expects
                unnormalized labels and handles the 1-5 → 0-4 index shift internally.
        """

        self.annotations = []
        self.annotation_keys = []
        with open(annotations_file, "r", encoding="utf-8") as fpr:
            for line in fpr:
                data_dict = json.loads(line)
                ratings = [r for r in data_dict["ratings"] if r is not None and 0 <= r <= 5]
                if len(ratings) > 0:
                    self.annotations.append(data_dict)
                    self.annotation_keys.append(data_dict["id"])

        self.prompts = prompts if prompts is not None else [""]
        self.context_prefix = "|context: "
        self.question_prefix = "|question: "
        self.answer_prefix = "|correct answer: "
        self.candidate_prefix = "|candidate answer: "
        self.transcript_prefix = "|transcription: "
        self.skip_rationale = skip_rationale
        self.skip_question = skip_question
        self.add_transcript = add_transcript
        self.log_fn = log_fn or print

        self.normalize_ratings = normalize_ratings

        if self.normalize_ratings:
            self.log_fn(
                "Normalizing ratings from [1, 5] to [0, 1] by subtracting 1 and dividing by 4."
            )

        if filter_func is not None:
            self.log_fn(f"Length before filtering: {len(self.annotations)}")
            self.annotation_keys = [
                key for idx, key in enumerate(self.annotation_keys) if filter_func(self[idx])
            ]
            # self.keys_to_id = {key: i for i, key in enumerate(self.annotation_keys)}
            # self.annotation_ids = ["_".join(key.split("_")[:-2]) for key in self.annotation_keys]
            # self.model_ids = ["_".join(key.split("_")[-2:]) for key in self.annotation_keys]
            self.log_fn(f"Length after filtering: {len(self.annotation_keys)}")

    def __len__(self):
        return len(self.annotation_keys)

    def keys(self):
        return self.annotation_keys

    def __getitem__(self, idx):
        item = self.annotations[idx]
        prompt = random.choice(self.prompts)
        context = item["rationale"]
        question = item["question"]
        answer = item["reference"]
        candidate_answer = item["candidate"]
        if self.add_transcript and "transcription" in item:
            context = f"{self.transcript_prefix}{item['transcription']} {context}"
        ratings = item["ratings"]
        if self.normalize_ratings:
            ratings = [
                float(rating - 1) / 4
                for rating in ratings
                if rating is not None and 0 <= rating <= 5
            ]
        else:
            ratings = [
                float(rating) for rating in ratings if rating is not None and 0 <= rating <= 5
            ]
        n_r = len(ratings)
        average_rating = sum(ratings) / n_r

        # unbiased estimator (Bessel's correction); 0.0 for singletons
        rating_variance = (
            sum((x - average_rating) ** 2 for x in ratings) / (n_r - 1) if n_r > 1 else 0.0
        )
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

        return {
            "text": format_free,
            "soft_labels": ratings,
            "average_rating": average_rating,
            "rating_variance": rating_variance,
            "n_ratings": n_r,
            "metadata": item,
            "prompt": prompt,
            "id": self.annotation_keys[idx],
        }

    def __iter__(self):
        for i in range(len(self.annotation_keys)):
            yield self[i]

    @property
    def seq_lengths(self):
        """Character lengths of each formatted text item (proxy for token length)."""
        prompt = self.prompts[0]
        lengths = []
        for i in range(len(self)):
            item = self.annotations[i]
            context = item["rationale"]
            if self.add_transcript and "transcription" in item:
                context = f"{self.transcript_prefix}{item['transcription']} {context}"
            text = prompt
            if not self.skip_rationale:
                text += f" {self.context_prefix}{context}"
            if not self.skip_question:
                text += f" {self.question_prefix}{item['question']}"
            text += f" {self.answer_prefix}{item['reference']}"
            text += f" {self.candidate_prefix}{item['candidate']}"
            lengths.append(len(text))
        return lengths


# Not used
# class InferenceDataset(UnifiedAnnotationDataset):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def __getitem__(self, idx):
#         item = self.annotations[self.annotation_keys[idx]]
#         prompt = random.choice(self.prompts)
#         context = item["rationale"]
#         question = item["question"]
#         answer = item["reference"]
#         candidate_answer = item["candidate"]
#         ratings = [0.0, 1.0]
#         average_rating = sum(ratings) / len(ratings)
#         rating_variance = sum((x - average_rating) ** 2 for x in ratings) / len(ratings)
#         format_free = (
#             f"{prompt} "
#             f"{self.context_prefix}{context} "
#             f"{self.question_prefix}{question} "
#             f"{self.answer_prefix}{answer} "
#             f"{self.candidate_prefix}{candidate_answer}"
#         )
#         model_id = self.model_ids[idx]
#         return {
#             "text": format_free,
#             "soft_labels": ratings,
#             "average_rating": average_rating,
#             "rating_variance": rating_variance,
#             "metadata": item,
#             "prompt": prompt,
#             "idx": self.annotation_keys[idx],
#             "annotation_id": self.annotation_ids[idx],
#             "model_id": model_id,
#         }


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

    @property
    def seq_lengths(self):
        """Character lengths of each item across all concatenated datasets."""
        return [length for ds in self.datasets for length in ds.seq_lengths]


class BucketBatchSampler(Sampler):
    """Batch sampler that groups sequences by length bucket to reduce padding waste.

    Indices within each bucket are shuffled, and bucket order is also shuffled
    each epoch, so training dynamics are preserved.
    """

    def __init__(self, lengths, batch_size, bucket_width=50, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.bucket_width = bucket_width
        self.shuffle = shuffle

    def __iter__(self):
        # Group indices into buckets by character-length range
        buckets: dict[int, list[int]] = {}
        for idx, length in enumerate(self.lengths):
            bucket_id = length // self.bucket_width
            buckets.setdefault(bucket_id, []).append(idx)

        # Shuffle within each bucket
        if self.shuffle:
            for indices in buckets.values():
                random.shuffle(indices)

        # Shuffle bucket order
        bucket_keys = list(buckets.keys())
        if self.shuffle:
            random.shuffle(bucket_keys)

        # Yield complete batches; drop last incomplete batch per bucket
        for key in bucket_keys:
            indices = buckets[key]
            for i in range(0, len(indices) - len(indices) % self.batch_size, self.batch_size):
                yield indices[i : i + self.batch_size]

    def __len__(self):
        buckets: dict[int, int] = {}
        for length in self.lengths:
            bucket_id = length // self.bucket_width
            buckets[bucket_id] = buckets.get(bucket_id, 0) + 1
        return sum(count // self.batch_size for count in buckets.values())


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
        all_n_ratings = [item["n_ratings"] for item in batch]
        # all_annotation_ids = [item["annotation_id"] for item in batch]
        # all_model_ids = [item["model_id"] for item in batch]
        all_idxs = [item["id"] for item in batch]

        tokenized = tokenizer(
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
        n_ratings = torch.tensor(all_n_ratings, dtype=torch.long)

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
                "n_ratings": n_ratings,
                "metadata": all_metadata,
                "prompt": all_prompts,
                # "annotation_id": all_annotation_ids,
                # "model_id": all_model_ids,
                "idx": all_idxs,
                **extra_kwargs,
            }
        )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_jsonl", type=str, default="data/stage3_human/seed_42/train.jsonl")
    parser.add_argument("--skip_question", action="store_true")
    parser.add_argument("--skip_rationale", action="store_true")
    parser.add_argument("--normalize_ratings", action="store_true")
    args = parser.parse_args()

    dataset = UnifiedAnnotationDataset(args.data_jsonl, normalize_ratings=args.normalize_ratings)
    for item in dataset:
        print(item)
        break
