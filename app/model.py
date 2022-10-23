from transformers import BertForSequenceClassification
from config import CONFIG

Model = BertForSequenceClassification.from_pretrained(
    CONFIG["MODEL_PATH"],
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False,
)