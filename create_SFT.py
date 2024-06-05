import time
import os
from random import randrange, sample, seed
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer
from tqdm import tqdm
from random import choices
import re
import random
import wandb
import json 
from tqdm import tqdm
from random import choices
import matplotlib.pyplot as plt
from datasets import load_dataset, DatasetDict, Dataset
from sklearn.model_selection import train_test_split
seed(42)


use_flash_attention2 = False

# change those lines to re-run in diff setup   and output_dir, trainer.save_model for different model
PIPELINE = 'topical_moderation'
dataset_path = "train_test_data/all_queries.txt"
TASK_LABELS = ['harmful', 'irrelevant', 'relevant'] #['helpful', 'harmful'] ['irrelevant', 'relevant']
model_id = "google/gemma-2b-it"   # model_id = "mistralai/Mistral-7B-Instruct-v0.3"


#dataset_path = "train_test_data/relevant_irrelevant_queries.txt"
#dataset_path = "train_test_data/harmful_helpful_queries.txt"

def build_instruction_dataset(file_path):
    instructions = []
    data_dict = []
    is_instruction = True  # Flag to determine whether we are reading instructions

    with open(file_path, 'r') as file:
        for line in file:
            stripped_line = line.strip()
               
            if stripped_line in TASK_LABELS:
                is_instruction = False

            if is_instruction:
                instructions.append(line)
            else:
                data_dict.append({
                    'prompt': ''.join(instructions), 
                    'completion': stripped_line
                
                })
                instructions = []
                is_instruction = True
                    
    return data_dict



def write_to_jsonl(data, file_path):
    """
    Writes a list of dictionaries to a JSONL file.
    
    Parameters:
    - data: List of dictionaries to write to the file.
    - file_path: Path to the JSONL file.
    """
    with open(file_path, 'w') as file:
        for record in data:
            json_record = json.dumps(record)
            file.write(json_record + '\n')
     
    
data_dict = build_instruction_dataset(dataset_path)
write_to_jsonl(data=data_dict, file_path=os.path.basename(dataset_path).replace('.txt', '.jsonl')) 

# load jsonl dataset
dataset_mod = load_dataset("json", data_files=os.path.basename(dataset_path).replace('.txt', '.jsonl'), split="train")


prompts_train, prompts_test, completions_train, completions_test = train_test_split(
    dataset_mod['prompt'], dataset_mod['completion'], test_size=0.3, random_state=42
)

# Create train and test datasets by combining prompts and completions
train_dataset = Dataset.from_dict({"prompt": prompts_train, "completion": completions_train})
test_dataset = Dataset.from_dict({"prompt": prompts_test, "completion": completions_test})


write_to_jsonl(train_dataset, f"{os.path.basename(dataset_path).replace('.txt', '')}_train.jsonl")
write_to_jsonl(test_dataset, f"{os.path.basename(dataset_path).replace('.txt', '')}_test.jsonl")


# BitsAndBytesConfig int-4 config 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if use_flash_attention2 else torch.float16
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    quantization_config=bnb_config, 
    use_cache=False, 
    device_map="auto",
    token = "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", # if model is gated like llama or mistral
    attn_implementation="flash_attention_2" if use_flash_attention2 else "sdpa"
)
model.config.pretraining_tp = 1

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    token = "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", # if model is gated like llama or mistral
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj", 
            "up_proj", 
            "down_proj",
        ]
)

# Prepare model for training
model = prepare_model_for_kbit_training(model)

args = TrainingArguments(
    output_dir=f"{PIPELINE}_gemma-sft-trainer",
    num_train_epochs=1,
    per_device_train_batch_size=6 if use_flash_attention2 else 2, # you can play with the batch size depending on your hardware
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=2e-4,
    bf16=use_flash_attention2,
    fp16=not use_flash_attention2,
    tf32=use_flash_attention2,
    max_grad_norm=0.3,
    warmup_steps=5,
    lr_scheduler_type="linear",
    disable_tqdm=False,
    report_to="none"
)

model = get_peft_model(model, peft_config)

def formatting_prompts_func(example):
    text = f"### Question: {example['prompt']}\n ### Answer: {example['completion']}"
    
    return text


trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
    max_seq_length=2048,
    tokenizer=tokenizer,
    packing=True,
    args=args,
    formatting_func=formatting_prompts_func
)

trainer.train()

trainer.save_model(output_dir = f"{PIPELINE}_gemma-sft-trainer")

