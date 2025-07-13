````markdown
# MedGPT-QLoRA

> Fine-tuning LLaMA 2 on a medical Q&A dataset using QLoRA in Google Colab

---

## Overview

**MedGPT-QLoRA** is a lightweight, resource-efficient fine-tuning pipeline for adapting [LLaMA 2 7B Chat](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf) to the medical domain. Using QLoRA and 4-bit quantization via `bitsandbytes`, this project enables fine-tuning on modest hardware (e.g., Google Colab T4 GPU) with excellent performance on health-related queries.

---

## Dataset

- **Name**: [HealthCareMagic-100k](https://www.kaggle.com/datasets/gunman02/health-care-magic)  
- **Source**: Kaggle  
- **Description**: A collection of patient health queries and expert answers.

---

## Features

- Fine-tunes LLaMA 2 7B Chat with `SFTTrainer` from Hugging Face's TRL library  
- Uses LoRA for parameter-efficient tuning via the `peft` library  
- Runs on consumer GPUs using 4-bit quantization with `bitsandbytes`  
- Tailors the model for medical assistant-style conversations

---

## Tech Stack

- [Transformers](https://github.com/huggingface/transformers)  
- [PEFT (LoRA)](https://github.com/huggingface/peft)  
- [TRL (SFTTrainer)](https://github.com/huggingface/trl)  
- [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes)  
- Google Colab (T4 GPU)

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/medgpt-qlora.git
cd medgpt-qlora
````

### 2. Install dependencies

```bash
pip install -q kagglehub opendatasets accelerate==0.21.0 peft==0.4.0 transformers==4.31.0 trl==0.4.7
pip install bitsandbytes
```

> Or run the Colab notebook if you're using Google Colab

### 3. Run training

```bash
python train.py
```

### 4. Generate predictions

```bash
python inference.py
```

---

## Example Prompt

```text
Within the past few hours, I've developed a persistent dry cough, low-grade fever, and mild chest discomfort. I'm generally healthy with no chronic conditions. Could these be early signs of pneumonia or something else?
```

**Model Output** (sample):

> "The symptoms you're describing may be early signs of a respiratory infection, possibly viral in origin..."

---

## Project Structure

```text
├── train.py                  # Full training pipeline with QLoRA  
├── inference.py              # Inference using the fine-tuned model  
├── llama2_chat_formatted_data.csv  
├── requirements.txt  
└── README.md
```

---

## Authentication

To load gated models like LLaMA 2 from Hugging Face:

1. Create a token at: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Store it as `HF_TOKEN` in your environment or Colab secrets

---

## Credits

* [NousResearch](https://huggingface.co/NousResearch) for the base LLaMA-2 model
* [HealthCareMagic Dataset](https://www.kaggle.com/datasets/gunman02/health-care-magic)
* Hugging Face 🤗 ecosystem (Transformers, TRL, PEFT)

---

## License

This project is for educational and research purposes only. Always consult licensed professionals for real medical advice.

```


```
