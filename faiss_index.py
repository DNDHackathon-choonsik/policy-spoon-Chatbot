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

combined_texts = [
    item['text'] for item in data['policies']
]

titles = [
    item.get('polyBizSjnm', 'No Title') for item in data['policies']
]

data_dict = {
    'title': titles,
    'text': combined_texts
}

dataset = Dataset.from_dict(data_dict)

tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq", use_auth_token=huggingface_token)
model = RagModel.from_pretrained("facebook/rag-sequence-nq", use_auth_token=huggingface_token)

def embed(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    with torch.no_grad():
        outputs = model.question_encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        embeddings = outputs[0].squeeze().cpu().numpy()
    return embeddings

sample_text = "sample text"
sample_vector = embed(sample_text)
print(f"Sample embedding shape: {sample_vector.shape}")

dimension = sample_vector.shape[0]
index = faiss.IndexFlatL2(dimension)

vectors = np.vstack([embed(text) for text in combined_texts])
index.add(vectors)

faiss.write_index(index, "custom_faiss_index")

dataset_with_embeddings = {
    'title': titles,
    'text': combined_texts,
    'embeddings': [embed(text).tolist() for text in combined_texts]
}

for i, emb in enumerate(dataset_with_embeddings['embeddings']):
    print(f"Embedding {i} shape: {np.array(emb).shape}")

features = Features({
    'title': Value('string'),
    'text': Value('string'),
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
