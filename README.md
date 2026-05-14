# CodeLens Research 🔬

> **Code → Research Paper Generator**  
> Converts source code into structured, publication-ready research paper sections using a fine-tuned CodeT5+ / T5 seq2seq pipeline.



## Screenshots

<img width="2800" height="1556" alt="image" src="https://github.com/user-attachments/assets/c503d5b9-5ea1-4fc3-a5b9-90cbf051850d" />

<img width="2338" height="1172" alt="image" src="https://github.com/user-attachments/assets/fa8fad38-fdbc-4e43-82f6-78dc99b6dfd3" />





## What It Does

Paste in a code snippet and get back a full academic paper draft:

| Section | Description |
|---|---|
| Abstract | Concise 150-200 word summary |
| Introduction | Background, motivation, objectives |
| Methodology | System design & approach |
| Results | Performance analysis |
| Conclusion | Findings & future directions |



## Architecture

```
Code Input
    │
    ▼
CodeT5+ Encoder  ←  understands code syntax & semantics
    │
    ▼
T5-large Decoder  ←  generates fluent academic English
    │
    ▼
Paper Sections (Abstract · Introduction · Methodology · Results · Conclusion)
```

**Stack:** HuggingFace Transformers · PyTorch · scikit-learn · ROUGE evaluation

---

## Setup

```bash
git clone https://github.com/yourusername/codelens-research
cd codelens-research
pip install -r requirements.txt
```

Requires Python 3.9+. GPU recommended but not required.

---

## Usage

### Quick Demo
```bash
python demo.py
```

### Generate a Paper from Your Code
```python
from model import CodeToPaperModel

model = CodeToPaperModel()
paper = model.generate_paper(your_code_string)
print(model.format_paper(paper, title="My Research Paper"))
```

### Fine-tune on Your Own Data
```python
from model import CodeToPaperModel
from dataset import CodePaperDataset

model = CodeToPaperModel()

# your_data = [{"code": "...", "abstract": "...", "methodology": "..."}, ...]
dataset = CodePaperDataset(model.tokenizer).from_list(your_data)
model.train(dataset)
```

**Data format** — a JSON list of objects:
```json
[
  {
    "code": "def bubble_sort(arr): ...",
    "abstract": "We present ...",
    "introduction": "Sorting algorithms ...",
    "methodology": "The algorithm iterates ...",
    "results": "Benchmarks show ...",
    "conclusion": "In this paper ..."
  }
]
```

## Evaluation

ROUGE-1/2/L scores are computed automatically during training and can be run manually:

```python
from evaluate import PaperEvaluator

evaluator = PaperEvaluator()
scores = evaluator.evaluate(generated_paper, reference_paper)
```



## Project Structure

```
codelens-research/
├── config.py       # All hyperparameters & settings
├── dataset.py      # Data loading, cleaning, tokenisation
├── model.py        # CodeT5+ seq2seq model, training & inference
├── evaluate.py     # ROUGE + readability evaluation
├── demo.py         # End-to-end demo script
├── requirements.txt
└── README.md
```



## Configuration

All settings live in `config.py`. Key options:

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `codet5-base` | Base pretrained model |
| `max_input_length` | 512 | Max code tokens |
| `max_target_length` | 512 | Max output tokens |
| `num_train_epochs` | 5 | Training epochs |
| `num_beams` | 5 | Beam search width |
| `learning_rate` | 3e-4 | Adam learning rate |

