# This file runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model
from sentence_transformers import SentenceTransformer

def download_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # do a dry run of loading the huggingface model, which will download weights
    model = SentenceTransformer(
        model_name_or_path='sentence-transformers/distiluse-base-multilingual-cased-v2',
        device = device,
        cache_folder='model'
    )
    # dry run for download
    model.encode(['hello'])

if __name__ == "__main__":
    download_model()