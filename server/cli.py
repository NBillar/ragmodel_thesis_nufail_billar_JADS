import os
from dotenv import load_dotenv
import logging
from RAGHelper_local import RAGHelperLocal


# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Choose the appropriate helper based on environment variables

logger.info("Instantiating the local RAG helper.")
raghelper = RAGHelperLocal(logger)

# Ensure data is loaded
raghelper.load_data()
import torch
print("MPS available:", torch.backends.mps.is_available())
# Start a simple CLI loop
print("Welcome to the RAG system CLI ya akhi. Type your question and press Enter (type 'exit' to quit).")
history = []  # simplified memory / awareness of CLI

while True:
    user_query = input("Your question: ")
    if user_query.strip().lower() == "exit":
        break

    # Append the previous assistant response to the history, if any
    new_history, response = raghelper.handle_user_interaction(user_query, history)
    answer = raghelper._extract_reply(response)

    if answer:
        history.append({"role": "user", "content": user_query})
        history.append({"role": "assistant", "content": answer})
        print("Assistant:", answer)
    else:
        print("Assistant: No answer returned")