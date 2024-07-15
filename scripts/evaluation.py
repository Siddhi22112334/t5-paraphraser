import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from datasets import load_metric, Dataset
import pandas as pd

# Load the dataset from a CSV file
def load_and_prepare_dataset(csv_path):
    df = pd.read_csv(csv_path)
    dataset = Dataset.from_pandas(df)
    
    # Map the dataset to the correct format
    def preprocess(example):
        return {
            'input_text': 'paraphrase: ' + example['original sentence'],
            'target_text': example['Paraphrased sentence']
        }
    
    dataset = dataset.map(preprocess, remove_columns=['original sentence', 'Paraphrased sentence'])
    return dataset

def check_device():
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Using device: {device}")
    return device

def main(args):
    print("Loading evaluation dataset...")
    eval_dataset = load_and_prepare_dataset(args.eval_csv_path)

    print("Loading model and tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_path)

    def tokenize(batch):
        input_texts = batch['input_text']
        target_texts = batch['target_text']
        input_encodings = tokenizer(input_texts, truncation=True, padding='max_length', max_length=128)
        target_encodings = tokenizer(target_texts, truncation=True, padding='max_length', max_length=128)

        encodings = {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }
        return encodings

    print("Tokenizing evaluation dataset...")
    tokenized_eval_dataset = eval_dataset.map(tokenize, batched=True)

    # Load the BLEU metric
    bleu = load_metric('sacrebleu')

    # Define the compute_metrics function
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Replace -100 in the labels as we can't decode them
        decoded_labels = [[label] for label in decoded_labels]

        result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": result["score"]}

    training_args = TrainingArguments(
        per_device_eval_batch_size=4,
        dataloader_drop_last=False,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Evaluate the model
    print("Evaluating the model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

if __name__ == "__main__":
    eval_csv_path = "path/to/your/eval_dataset.csv"
    model_path = "path/to/your/trained_model"
    model_name = 't5-small'

    class Args:
        pass

    args = Args()
    args.eval_csv_path = eval_csv_path
    args.model_path = model_path
    args.model_name = model_name
    args.device = check_device()

    main(args)
