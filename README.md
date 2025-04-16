# Hallucination Detection in Language Model Generations

This project is based on the works of [HaloScope: Harnessing Unlabeled LLM Generations for Hallucination Detection](https://github.com/deeplearning-wisc/haloscope)

This project proposes a method (Haloscope) for training a hallucination detector using an unlabeled dataset of prompts and LLM-generated answers. By analyzing the latent activation space from which LLMs generate responses, the researchers aim to identify a subspace that captures the patterns associated with hallucinated outputs. This approach enables the automatic inference of labels—distinguishing hallucinated from truthful answers—based on structural properties of the latent space, allowing for scalable hallucination detection without manual annotation.

---

## Installation Instructions

To install the main libraries required for running the project:

```bash
pip install git+https://github.com/davidbau/baukit
pip install git+https://github.com/lucadiliello/bleurt-pytorch.git
pip install huggingface_hub
pip install evaluate
pip install t5
```


You will also need a HuggingFace read token to download the Llama 2 and BLUERT models.

---

## Project Structure

```
.
├── src/                  # Stores generated answers and hidden states
│   ├── tqa_hal_det/               # Dataset-specific subdirectory for TruthfulQA
│       ├── answers/              # Generated model responses (.npy)
│       ├── most_likely_*.npy     # Extracted hidden state features (layer-wise)
├── ml_tqa_bleurt_score.npy        # BLEURT-based hallucination score labels
├── lstm_hallucination.pt          # Trained LSTM classifier checkpoint

```

---

## Utility files from other projects

| File                  | Purpose |
|-----------------------|---------|
| `ProjectCode.ipynb`   | Main training and evaluation pipeline for both classifiers |


---

## 📊 Code Outputs

This project generates the following files and directories during execution:

- `save_for_eval/.../answers/` — Saved `.npy` arrays containing model-generated answers

---

## 📈 Example Evaluation Output

```
Epoch 20 - Loss: 0.3721 - Val AUROC: 0.768
Test AUROC (LSTM): 0.782
Test AUROC (Linear Probe): 0.661
```

---

## 📝 Notes

- Classifiers 

---
