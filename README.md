# rag-agent-example

## Steps to run
* Create embeddings using all the data available in the knowledge base:
     `python vector_db.py`
* Update the prompt information and use the embeddings as a knowledge base in conversable.py. Execute the code to get the final scores predicted by the model:
     `python conversable.py`
