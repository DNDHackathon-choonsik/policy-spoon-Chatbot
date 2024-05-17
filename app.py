import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import RagTokenizer, RagRetriever, RagModel, AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import torch
import numpy as np
import atexit
from datasets import load_from_disk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')

# Initialize FastAPI
app = FastAPI()

# Load RAG tokenizer, retriever, and model
rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq", token=huggingface_token, force_download=True)
rag_retriever = RagRetriever.from_pretrained("custom_rag_retriever", token=huggingface_token, force_download=True)
rag_model = RagModel.from_pretrained("facebook/rag-sequence-nq", token=huggingface_token, force_download=True)
phi_tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct", token=huggingface_token, trust_remote_code=True, force_download=True)
phi_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct", token=huggingface_token, trust_remote_code=True,
    attn_implementation='eager',
    force_download=True)

# Load dataset with embeddings
dataset_path = "custom_dataset_with_embeddings"
dataset = load_from_disk(dataset_path)

# Move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rag_model = rag_model.to(device)
phi_model = phi_model.to(device)

class Query(BaseModel):
    question: str

def postprocess_output(text):
    # Ensure the generated text ends at a complete sentence.
    end_punctuation = {".", "!", "?"}
    for i in range(len(text) - 1, -1, -1):
        if text[i] in end_punctuation:
            return text[:i + 1]
    return text

@app.post("/generate")
def generate_answer(query: Query):
    try:
        logger.info("Received question: %s", query.question)

        # Tokenize the input question using the RAG tokenizer
        inputs = rag_tokenizer(query.question, return_tensors="pt").to(device)
        input_ids = inputs.input_ids

        logger.info("Tokenized input_ids: %s", input_ids)

        # Encode the question to get hidden states
        question_hidden_states = rag_model.question_encoder(input_ids)[0]

        logger.info("Encoded question hidden states: %s", question_hidden_states.shape)

        # Retrieve context documents using RAG retriever
        question_hidden_states_np = question_hidden_states.detach().cpu().numpy()
        question_input_ids_np = input_ids.detach().cpu().numpy()

        logger.info("Calling retriever with question_input_ids: %s and question_hidden_states: %s", question_input_ids_np, question_hidden_states_np)

        retrieved_docs = rag_retriever(question_input_ids=question_input_ids_np, question_hidden_states=question_hidden_states_np)

        logger.info("Retrieved documents: %s", retrieved_docs)

        # Verify the contents of retrieved_docs
        if not retrieved_docs or 'doc_ids' not in retrieved_docs:
            raise ValueError("No documents retrieved or 'doc_ids' missing in retrieved documents.")

        # Decode the retrieved documents
        doc_ids = retrieved_docs['doc_ids']
        if doc_ids.size == 0 or np.all(doc_ids == -1):
            raise ValueError("No valid document IDs found in retrieved documents.")

        # Get only the first valid document for simplicity
        valid_doc_ids = doc_ids[doc_ids != -1]
        if valid_doc_ids.size == 0:
            raise ValueError("No valid document IDs found in retrieved documents.")

        doc_id = valid_doc_ids[0]
        doc = dataset[int(doc_id)]  # Get document from dataset

        if not doc:
            raise ValueError(f"Document with ID {doc_id} not found.")

        retrieved_text = doc['text']

        # Prepare the input for Phi model
        phi_input_text = f"Context: {retrieved_text}\n\nQuestion: {query.question}\n\nAnswer:"
        phi_input_ids = phi_tokenizer(phi_input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device)

        logger.info("Context input for Phi model: %s", phi_input_text)

        # Generate the answer using the Phi model
        with torch.no_grad():
            outputs = phi_model.generate(phi_input_ids, num_return_sequences=1, num_beams=3, max_length=150, eos_token_id=phi_tokenizer.eos_token_id)
        answer = phi_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove context and question from the answer
        answer = answer.replace(phi_input_text, "").strip()

        # Ensure the answer ends with a complete sentence
        answer = postprocess_output(answer)

        logger.info("Generated answer: %s", answer)

        return {"answer": answer}

    except ValueError as e:
        logger.warning("RAG retriever failed, falling back to language model: %s", str(e))

        # Directly use the Phi model for answering
        phi_input_ids = phi_tokenizer(query.question, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device)

        with torch.no_grad():
            outputs = phi_model.generate(phi_input_ids, num_return_sequences=1, num_beams=3, max_length=150, eos_token_id=phi_tokenizer.eos_token_id)
        answer = phi_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Ensure the answer ends with a complete sentence
        answer = postprocess_output(answer)

        logger.info("Generated answer using language model: %s", answer)

        return {"answer": answer.strip()}

    except Exception as e:
        logger.exception("Error generating answer")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

def clean_up_resources():
    logger.info("Cleaning up resources...")
    torch.cuda.empty_cache()

# Register cleanup function
atexit.register(clean_up_resources)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
