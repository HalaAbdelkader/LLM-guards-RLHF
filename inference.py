from datasets import load_dataset
import json
import transformers
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel
import torch 


if __name__ == '__main__':
    dataset_mod = load_dataset("json", data_files="relevant_irrelevant_queries_test.jsonl", split="train")
    print(dataset_mod)
    
#     model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
#     pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
    
#     pipeline("Hey how are you doing today?")
    
#     quit(0)
#     # Define the path to your trained SFT model directory
#     model_path = "facebook/opt-350m" #"llama3-sft-trainer"

#     # Load the SFT model for sequence generation
#     device = "cuda:0"
#     model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    base_model_name = "meta-llama/Meta-Llama-3-8B" 
    adapter_model_name = "llama3-sft-trainer"

    model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, adapter_model_name, torch_dtype=torch.bfloat16, device_map="auto")
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    
    while True:
        index = input("Enter index: ")
        # Example prompt for text generation
        prompt = f"### Question: {dataset_mod[int(index)]['prompt']}\n ### Answer: "
        print(prompt)
        
        inputs = tokenizer.encode(prompt, max_length=1024, return_tensors="pt").to(model.device)
        outputs = model.generate(inputs, max_length=1024)
        print("generation \n")
        print(tokenizer.decode(outputs[0]))
