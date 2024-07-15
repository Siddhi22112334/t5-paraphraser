import os
import time
import torch
import logging
import argparse
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments, TrainerCallback
from datasets import load_metric, Dataset
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SaveCheckpointCallback(TrainerCallback):
    def __init__(self, trainer, save_interval=600):  # Save checkpoint every 600 seconds (10 minutes)
        self.trainer = trainer
        self.save_interval = save_interval
        self.last_save_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if time.time() - self.last_save_time >= self.save_interval:
            self.last_save_time = time.time()
            output_dir = os.path.join(args.output_dir, f'checkpoint-{state.global_step}')
            os.makedirs(output_dir, exist_ok=True)
            # Save model and state
            self.trainer.save_model(output_dir)
            state.save_to_json(os.path.join(output_dir, 'trainer_state.json'))
            torch.save(self.trainer.optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))
            torch.save(self.trainer.lr_scheduler.state_dict(), os.path.join(output_dir, 'scheduler.pt'))
            logger.info(f"Checkpoint saved at step {state.global_step} to {output_dir}")

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
    logger.info(f"Using device: {device}")
    return device

def main(args):
    logger.info("Loading datasets...")
    train_dataset = load_and_prepare_dataset(args.train_csv_path)
    eval_dataset = load_and_prepare_dataset(args.eval_csv_path)

    logger.info("Loading model and tokenizer...")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

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

    logger.info("Tokenizing datasets...")
    tokenized_train_dataset = train_dataset.map(tokenize, batched=True)
    tokenized_eval_dataset = eval_dataset.map(tokenize, batched=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=3,
        logging_dir='./logs',
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        fp16=True if args.device == 'cuda' else False,  # Enable fp16 only for CUDA
    )

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

    # Check if there's a valid checkpoint
    last_checkpoint = None
    if os.path.isdir(args.output_dir):
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
            last_checkpoint = os.path.join(args.output_dir, checkpoints[-1])
            if not os.path.exists(os.path.join(last_checkpoint, 'trainer_state.json')):
                last_checkpoint = None

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        compute_metrics=compute_metrics,
    )

    # Add the callback to the trainer
    callback = SaveCheckpointCallback(trainer, save_interval=600)
    trainer.add_callback(callback)

    # Train the model
    if last_checkpoint:
        logger.info(f"Resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        logger.info("No checkpoint found. Starting training from scratch.")
        trainer.train()

    # Save the model
    logger.info("Saving the model...")
    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)

    # Evaluate the model
    logger.info("Evaluating the model...")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_results}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a T5 model for paraphrasing.")
    parser.add_argument('--train_csv_path', type=str, required=True, help="Path to the training dataset CSV file.")
    parser.add_argument('--eval_csv_path', type=str, required=True, help="Path to the evaluation dataset CSV file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save checkpoints and model outputs.")
    parser.add_argument('--save_model_path', type=str, required=True, help="Path to save the final model.")
    parser.add_argument('--model_name', type=str, default='t5-small', help="Model name or path to a pre-trained model.")
    args = parser.parse_args()

    args.device = check_device()
    main(args)
