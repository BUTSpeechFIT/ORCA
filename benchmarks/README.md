# Corrected MMAU and MMAR Benchmarks

This directory contains corrected release versions of the MMAU test-mini benchmark and the MMAR benchmark. The corrections were made with the help of human annotators, who reviewed the audio questions, answer choices, and answers for accuracy and clarity.

The corrected files are:

- `MMAU_test_mini_open_ended_corrected.jsonl`
- `MMAR_open_ended_corrected.jsonl`

## What was corrected

Several items were corrected. Here are a few examples:

### MMAU

```
id: "72fb5481-73ae-409d-8e16-c94ac48d2ee4"
question: "Based on the given audio, identify the source of the speech."
choices: ["A child", "A woman and a child", "A woman", "A man"]
answer: "A woman and a child"
correction:
  old_answer: "A woman"
  new_answer: "A woman and a child"
```

```
id: "09247cc2-fb6a-43e0-ab58-e0c3f80a789b"
question: "How many times did the dog bark sound appear?"
choices: ["1", "2", "3", "4"]
answer: "2"
correction:
  old_answer: "1"
  new_answer: "2"
```

### MMAR

```
id: "BV1gk4y1R7wH_00-00-10_00-00-35"
question: "What is the most likely a scenario represented by the audio?"
choices: ["Celebration ceremony", "Garden activity", "Concert", "Sports competition"]
answer: "Sports competition"
correction:
  old_answer: "Celebration ceremony"
  new_answer: "Sports competition"
```

```
id: "BV19M4y1j764_00-01-03_00-01-23"
question: "Which country is this girl from"
choices: ["Australia", "Britain", "America", "Canada"]
answer: "Britain"
correction:
  old_answer: "British"
  new_answer: "Britain"
```

## Open-ended evaluation format

Many original benchmark questions are formulated as open-ended questions, but their answer choices are embedded in the question text or supplied alongside it. For example, a question may ask whether a sound came from a radio, fire truck, construction site, or airplane.

Embedding the choices in the question is an intentional reformulation that makes these benchmark items suitable for open-ended model evaluation. It is not itself a correction to the underlying benchmark annotation. The corrected files retain the answer choices and the corresponding reference answer so that models can respond freely while evaluation remains well-defined.

```text
id: "da2d42eb-b544-44dc-a507-0acf0bbb8d95"
original_question: "Based on the given audio, identify the source of the church bells."
open_ended_question: "Based on the audio, what is the source of the bells: a church, a school, a clock tower, or a fire station?"
choices: ["Church", "School", "Clock Tower", "Fire Station"]
answer: "Church"
```

## Citations

### MMAU

> *MMAU: A Massive Multi-Task Audio Understanding and Reasoning Benchmark.* Original benchmark paper. See the [MMAU project](https://github.com/SakshiAgarwal2025/MMAU) for the benchmark and publication details.

### MMAR

```bibtex
@inproceedings{mmar2025,
  title     = {MMAR: A Challenging Benchmark for Deep Reasoning in Speech, Audio, Music, and Their Mix},
  booktitle = {Proceedings of the International Conference on Multimedia and Expo Workshops},
  year      = {2025},
  doi       = {10.52202/085713-2252}
}
```

### ORCA

```bibtex
@article{sedlacek-etal-2026-orca,
    author = {Sedláček, Šimon and Barahona, Sara and Bolaños, Cecilia and Herrera-Alarcón, Laura and Udupa, Sathvik and López, Fernando and Ferner, Allison and Yusuf, Bolaji and Lozano-Diez, Alicia and Kesiraju, Santosh and Duraiswami, Ramani and Černocký, Jan},
    title = {ORCA: Open-ended Response Correctness Assessment for Audio Question Answering},
    journal = {Transactions of the Association for Computational Linguistics},
    volume = {14},
    pages = {2213--2233},
    year = {2026},
    doi = {10.1162/TACL.a.798},
    url = {https://doi.org/10.1162/TACL.a.798}
}
```
