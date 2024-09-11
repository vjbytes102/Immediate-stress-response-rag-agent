# Conversational agent with RAG
#https://github.com/grasool/Local-RAG-Chatbot
from openai import OpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

prompt_content = """
<User Prompt>
"""

def main():

    # Point to the local server
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="../embeddings/chroma_db_nccn", embedding_function=embedding_function)


    history = [
        {"role": "system", "content": '<System prompt>'},
        {"role": "user", "content": prompt_content},
    ]

    while True:
        completion = client.chat.completions.create(
            model="local-model",
            messages=history,
            temperature=0.7,
            stream=True,
        )

        new_message = {"role": "assistant", "content": ""}
        
        for chunk in completion:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
                new_message["content"] += chunk.choices[0].delta.content

        history.append(new_message)
        
        
        next_input = input("> ")
        search_results = vector_db.similarity_search(next_input, k=2)
        some_context = ""
        for result in search_results:
            some_context += result.page_content + "\n\n"
        history.append({"role": "user", "content": some_context + next_input})

if __name__ == "__main__":
    main()