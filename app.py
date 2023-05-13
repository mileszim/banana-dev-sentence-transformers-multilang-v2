from potassium import Potassium, Request, Response

from json import JSONEncoder
import json
import numpy
from sentence_transformers import SentenceTransformer
import torch

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

app = Potassium("embedding-multilang")

# @app.init runs at startup, and loads models into the app's context
@app.init
def init():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer(
        model_name_or_path='sentence-transformers/distiluse-base-multilingual-cased-v2',
        device=device,
        cache_folder='model'
    )
   
    context = {
        "model": model
    }

    return context

# @app.handler runs for every call
@app.handler()
def handler(context: dict, request: Request) -> Response:
    model = context.get("model")

    # get attributes
    sentences = request.json.get("sentences")
    batch_size = request.json.get("batch_size")
    batch_size = int(batch_size) if batch_size else 64

    # make it array if its not
    if not isinstance(sentences, list):
        sentences = [sentences]
    
    # get embeddings
    embeddings = model.encode(
        sentences=sentences,
        batch_size=batch_size
    )

    # format response
    res = {'sentences': sentences, 'embeddings': embeddings}

    # return response
    return Response(
        json = json.dumps(res, cls=NumpyArrayEncoder), 
        status=200
    )

if __name__ == "__main__":
    app.serve()