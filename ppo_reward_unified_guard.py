import re
import random
import torch
import wandb
import time
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from random import choices
import matplotlib.pyplot as plt
tqdm.pandas()
from colorama import Fore, Back, Style
from datasets import load_dataset
from datasets import Dataset 
from transformers import AutoTokenizer, pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer

use_flash_attention2 = False

MODEL = 'gemma' # mistral, llama, gemma 
PIPELINE = 'topical_moderation'
dataset_path = 'train_test_data/harmful_relevant_irrelevant_bank.csv'
ctrl_str = ["[Harmful]", "[Irrelevant]", "[Relevant]"]



bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if use_flash_attention2 else torch.float16
)

def extract_pipe_output(outputs):
    positive_logits = []
    for out in outputs:
        for element in out:
            if element["label"] == "LABEL_2":
                positive_logits.append(torch.tensor(element["score"]))
    return positive_logits


def pos_logit_to_reward(logit, task):
    """
    Take the positive sentiment logit and scale it for the task.
        task [negative]: reward = -logit
        task [neutral]: reward = -0.4*abs(logit) 
        task [positive]: reward = logit
    """
    for i in range(len(logit)):
        if task[i] == "[Harmful]":
            logit[i] = -logit[i]
        elif task[i] == "[Irrelevant]":
            logit[i] = -0.4 * torch.abs(logit[i]) 
        elif task[i] == "[Relevant]":
            pass
        else:
            raise ValueError("task has to be in [0, 1, 2]!")
    return logit


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def main():
    moderation_pipe_kwargs = {"top_k": None, "function_to_apply": "none"}

    config = PPOConfig(
        steps=51200, learning_rate=1.41e-5, remove_unused_columns=False, 
        backward_batch_size=8, global_backward_batch_size=8,
        global_batch_size=8,mini_batch_size=8,batch_size=8,
    )

    txt_in_len = 5
    txt_out_len = 20
    seed = 1    
    model_path = f"{PIPELINE}_{MODEL}-sft-trainer"
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Directory '{model_path}' does not exist. \n Please run create_SFT.py to create the fine tuned model")
    else:
        
        model = AutoModelForCausalLMWithValueHead.from_pretrained(model_path,low_cpu_mem_usage=True,torch_dtype=torch.float16,quantization_config=bnb_config)
        model_tokenizer = AutoTokenizer.from_pretrained(model_path)
        model_ref = create_reference_model(model)
        model_tokenizer.pad_token = model_tokenizer.eos_token
       
        # Three labels 
        df = pd.read_csv(dataset_path, encoding="ISO-8859-1")

        dataset = Dataset.from_pandas(df)

        dataset = dataset.map(
        lambda x: {"input_ids": model_tokenizer.encode(" " + x["Queries"], return_tensors="pt")[0, :txt_in_len]},
        batched=False,
        )
        dataset = dataset.map(lambda x: {"query": model_tokenizer.decode(x["input_ids"])}, batched=False)
        dataset.set_format("pytorch")

        config.query_dataset = dataset
        config.reward_model = f"{PIPELINE}_distilbert"

        print(f"\n The current config for the PPO pipeline is \n {config}")

        moderation_pipe_kwargs = {"top_k": None, "function_to_apply": "none"}

        ppo_trainer = PPOTrainer(config, model, model_ref, model_tokenizer, dataset, data_collator=collator)


        if ppo_trainer.accelerator.num_processes == 1:
            device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
        else:
            device = ppo_trainer.accelerator.device

        classifier_path = f"{PIPELINE}_distilbert"
        if not os.path.isdir(classifier_path):
            raise FileNotFoundError(f"Directory '{classifier_path}' does not exist. \n Please run bert_classifier_train.py to create the {PIPELINE} bert")
        else:
            moderation_pipe = pipeline("text-classification", classifier_path, device=device)
            text = "I have received my card, can you help me put it in the app?"
            output = moderation_pipe(text, **moderation_pipe_kwargs)
            print(Fore.RED + f"\n The below text {text} ")
            print(Fore.YELLOW + f"Created the following output from the classifier \n {output} ")

            text = "how would I break a jackasses neck?"
            output = moderation_pipe(text, **moderation_pipe_kwargs)
            print(Fore.RED + f"\n The below text {text} ")
            print(Fore.YELLOW + f"Created the following output from the classifier \n {output} ")


            text = "What are some ways to clean hands without water?"
            output = moderation_pipe(text, **moderation_pipe_kwargs)
            print(Fore.RED + f"\n The below text {text} ")
            print(Fore.YELLOW + f"Created the following output from the classifier \n {output} ")


            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # this should be handled by accelerate


            ctrl_tokens = dict((s, model_tokenizer.encode(s, return_tensors="pt").squeeze().to(device)) for s in ctrl_str)

            print(f"Based on the reward function and labels an example logit is as follows: \n {pos_logit_to_reward(torch.Tensor([4, 4, 4]), ctrl_str)}")
            


            generation_kwargs = {
                "min_length": -1,
                "top_k": 0.0,
                "top_p": 1.0,
                "do_sample": True,
                "pad_token_id": model_tokenizer.eos_token_id,
                "max_new_tokens": txt_out_len,
                "eos_token_id": model_tokenizer.eos_token_id,
            }
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            for epoch in range(2):
                for batch in tqdm(ppo_trainer.dataloader):
                    game_data = {}

                    # Prepend a random control token
                    task_list = choices(ctrl_str, k=config.batch_size)
                    game_data["query"] = [t + q for t, q in zip(task_list, batch["query"])]
                    
                    # Ensure input_ids are torch.Tensors, move to the same device, and concatenate with control tokens
                    query_tensors = [torch.cat((ctrl_tokens[t].to(device), torch.tensor(input_ids, dtype=torch.long).to(device))) for t, input_ids in zip(task_list, batch["input_ids"])]

                    # Get response from model 
                    response_tensors = []
                    for query in query_tensors:
                        response = ppo_trainer.generate(query, **generation_kwargs)
                        response_tensors.append(response.squeeze()[-txt_out_len:])
                    game_data["response"] = [model_tokenizer.decode(r.squeeze()) for r in response_tensors]

                    # moderation analysis
                    texts = [q + r for q, r in zip(batch["query"], game_data["response"])]
                    logits = extract_pipe_output(moderation_pipe(texts, **moderation_pipe_kwargs))
                    rewards = pos_logit_to_reward(logits, task_list)

                    # Run PPO training
                    t = time.time()
                    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

                    # Log stats for each control string
                    for cs in ctrl_str:
                        key = "env/reward_" + cs.strip("[]")
                        stats[key] = np.mean([r.cpu().numpy() for r, t in zip(rewards, task_list) if t == cs])
                    ppo_trainer.log_stats(stats, game_data, rewards)
                    
                  
            os.makedirs(f"./{PIPELINE}_{MODEL}_ppo", exist_ok=True)
            model.save_pretrained(f"./{PIPELINE}_{MODEL}_ppo", push_to_hub=False)
            model_tokenizer.save_pretrained(f"./{PIPELINE}_{MODEL}_ppo", push_to_hub=False)
         

if __name__ == "__main__":
    main()
