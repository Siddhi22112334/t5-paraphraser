# T5 Paraphrasing 

This project trains a T5-small model to paraphrase sentences using the PAWS dataset and additional custom data.

## Dataset

The dataset used in this project is primarily from the PAWS dataset, available [here](https://www.kaggle.com/datasets/thedevastator/the-paws-dataset-for-paraphrase-identification).

## Project Structure

t5-paraphrasing/
│
├── data/
│ ├── train.csv # Your training dataset
│ └── validation.csv # Your validation dataset
│
├── models/
│ └── trained_model/ # Your trained model files
│
├── scripts/
│ ├── train.py # Your training script
│ └── evaluate.py # Your evaluation script
│
├── .gitignore
└── README.md



## Requirements

- Python 3.x
- PyTorch
- Transformers (Hugging Face)
- Pandas

Install the required packages:
```sh
pip install transformers datasets sacrebleu accelerate torch