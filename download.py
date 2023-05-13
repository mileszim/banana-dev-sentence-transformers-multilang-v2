# This file runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model
from sentence_transformers import SentenceTransformer

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    SentenceTransformer(
        model_name_or_path='sentence-transformers/distiluse-base-multilingual-cased-v2'
    ).load('distiluse-base-multilingual-cased-v2')

if __name__ == "__main__":
    download_model()