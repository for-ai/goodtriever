from transformers import AutoModelForSequenceClassification
from generation.knn_transformers import

model = AutoModelForSequenceClassification.from_pretrained(
        "gpt2"
    )
