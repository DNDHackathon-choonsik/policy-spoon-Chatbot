import os
import json
import numpy as np
import faiss
from transformers import RagTokenizer, RagModel, RagRetriever
from datasets import Dataset, Features, Value, Sequence
from dotenv import load_dotenv
import torch

load_dotenv()
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')

with open('youth_policy_dataset.json', 'r') as f:
    data = json.load(f)

data_dict = {
    'title': [item['polyBizSjnm'] for item in data['policies']],
    'text': [item['polyItcnCn'] for item in data['policies']],
    'category': [item['category'] for item in data['policies']],
    'support': [item['sporCn'] for item in data['policies']]
}

dataset = Dataset.from_dict(data_dict)

tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq", token=huggingface_token, trust_remote_code=True)
model = RagModel.from_pretrained("facebook/rag-sequence-nq", token=huggingface_token, trust_remote_code=True, attn_implementation='eager')

def embed(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        outputs = model.question_encoder(input_ids=inputs.input_ids)
        embeddings = outputs[0].squeeze().cpu().numpy()
    return embeddings

sample_text = "sample text"
sample_vector = embed(sample_text)
print(f"Sample embedding shape: {sample_vector.shape}")

dimension = sample_vector.shape[0]
index = faiss.IndexFlatL2(dimension)

vectors = np.vstack([embed(q) for q in data_dict['title']])  # Stack to 2D array
index.add(vectors)

faiss.write_index(index, "custom_faiss_index")

dataset_with_embeddings = {
    'title': data_dict['title'],
    'text': data_dict['text'],
    'category': data_dict['category'],
    'support': data_dict['support'],
    'embeddings': [embed(q).tolist() for q in data_dict['title']]  # Keep as 1D
}

for i, emb in enumerate(dataset_with_embeddings['embeddings']):
    print(f"Embedding {i} shape: {np.array(emb).shape}")

features = Features({
    'title': Value('string'),
    'text': Value('string'),
    'category': Value('string'),
    'support': Value('string'),
    'embeddings': Sequence(Value('float32'))
})

rag_dataset = Dataset.from_dict(dataset_with_embeddings, features=features)
rag_dataset.save_to_disk('custom_dataset_with_embeddings')

retriever = RagRetriever.from_pretrained(
    "facebook/rag-sequence-nq",
    index_name="custom",
    passages_path="custom_dataset_with_embeddings",
    index_path="custom_faiss_index",
    use_auth_token=huggingface_token
)
retriever.save_pretrained("custom_rag_retriever")
