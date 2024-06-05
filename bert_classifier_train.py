from tqdm import tqdm
import numpy as np


tqdm.pandas()

from datasets import load_dataset



from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_metric


def tokenize(examples):
    outputs = tokenizer(examples['text'], truncation=True)
    return outputs

def compute_metrics(eval_preds):
    metric = load_metric("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


PIPELINE = 'topical'

if __name__ == '__main__':

    # Overall dataset
    #data_mod = load_dataset('csv',data_files="train_test_data/harmful_relevant_irrelevant_bank.csv",encoding="ISO-8859-1")
    #data_mod = load_dataset('csv',data_files="train_test_data/harmful_helpful_train_set.csv",encoding="ISO-8859-1")
    data_mod = load_dataset('csv',data_files="train_test_data/relevant_irrelevant.csv",encoding="ISO-8859-1")

    data_mod = data_mod.rename_column("Queries", "text")
    data_mod = data_mod.rename_column("Judgement","label")

    data_mod = data_mod.class_encode_column("label")
    split = data_mod['train'].train_test_split(test_size=0.3, seed=42)

    print(f"The current feature set for data are \n {split['test'].features}")


    model_name = "distilbert-base-uncased"
    # For overall data 
    model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=3)
    #model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=2)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_ds = split.map(tokenize, batched=True)


    training_args = TrainingArguments(num_train_epochs=1,
                                      output_dir=f"{PIPELINE}_distilbert-mod",
                                      push_to_hub=False,
                                      per_device_train_batch_size=16,
                                      per_device_eval_batch_size=16,
                                      eval_strategy="epoch", 
                                      report_to="none")

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = Trainer(model=model, tokenizer=tokenizer,
                      data_collator=data_collator,
                      args=training_args,
                      train_dataset=tokenized_ds["train"],
                      eval_dataset=tokenized_ds["test"], 
                      compute_metrics=compute_metrics)

    trainer.train()
    trainer.save_model(output_dir=f"{PIPELINE}_distilbert")
