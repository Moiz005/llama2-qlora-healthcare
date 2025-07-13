# Import All the Required Libraries

import torch
import kagglehub
from kagglehub import KaggleDatasetAdapter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
import pandas as pd
import opendatasets as od
from datasets import Dataset

# Set Up the Model, Dataset, and QLoRA Parameters

model_name = "NousResearch/Llama-2-7b-chat-hf"
dataset_name = "gunman02/health-care-magic"
new_model = "Llama-2-7b-chat-medgpt"

# QLoRA parameters
lora_r = 64
lora_alpha = 16
lora_dropout = 0.1

# bitsandbytes parameters
use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

# TrainingArguments parameters
output_dir = "./results"
num_train_epochs = 1
fp16 = True
bf16 = False
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
weight_decay = 0.001
optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
max_steps = -1
warmup_ratio = 0.03
group_by_length = True
save_steps = 0
logging_steps = 25

# SFT parameters
max_seq_length = None
packing = False
device_map = {"":0}

# Loading the Dataset

od.download("https://www.kaggle.com/datasets/gunman02/health-care-magic")

df = pd.read_json("health-care-magic/HealthCareMagic-100k.json")

# Preprocessing Dataset

df.head()

df.dtypes

df.describe(include='all')

df.shape

df.columns

df.isnull().sum()

# Formatting dataset according to Llama 2 input format

def format_llama2_chat(row):
    return f"<s>[INST] <<SYS>>\n{row['instruction']}\n<</SYS>>\n\n{row['input']} [/INST] {row['output']}"

df['text'] = df.apply(format_llama2_chat, axis=1)
df[['text']].to_csv("llama2_chat_formatted_data.csv", index=False, header=True)

df = pd.read_csv("llama2_chat_formatted_data.csv")
df.columns

df.dtypes

# Load Everything and Start Fine-Tuning

os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN"

dataset = Dataset.from_pandas(df)

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtye=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    r=lora_r,
    lora_dropout=lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
)

training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to="tensorboard"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

trainer.train()
trainer.model.save_pretrained(new_model)

# Use the Text Generation Pipeline

logging.set_verbosity(logging.CRITICAL)
prompt = (
    "Within the past few hours, I've developed a persistent dry cough, low-grade fever, "
    "and mild chest discomfort. I'm generally healthy with no chronic conditions. "
    "Could these be early signs of pneumonia or something else?"
)
formatted_prompt = f"<s>[INST] <<SYS>>\nYou are a knowledgeable and trustworthy medical assistant. Provide accurate and helpful answers.\n<</SYS>>\n\n{prompt} [/INST]"
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
result = pipe(formatted_prompt)
print(result[0]['generated_text'])