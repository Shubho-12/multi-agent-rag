from agents.retriever import RetrieverAgent
from agents.qa import QAgent
import os

# Load documents
data_folder = 'data'
docs = []
for filename in os.listdir(data_folder):
    if filename.lower().endswith('.txt'):
        with open(os.path.join(data_folder, filename), 'r', encoding='utf-8') as f:
            docs.append(f.read())

if not docs:
    print("No .txt files found in data/ — add sample.txt and try again.")
    exit(1)

# Initialize agents
retriever = RetrieverAgent()
retriever.add_documents(docs)  # Removed chunk_size & overlap
qa_agent = QAgent()

# Ask questions
while True:
    question = input("\nAsk a question (or type 'exit' to quit): ")
    if question.lower() == 'exit':
        break
    relevant_docs = retriever.retrieve(question, top_k=3)

    # Debug: show retrieved context
    print("\n--- Retrieved Context (truncated) ---")
    for i, d in enumerate(relevant_docs, 1):
        print(f"[{i}] {d[:400]}{'...' if len(d) > 400 else ''}\n")
    print("--- End Context ---\n")

    answer = qa_agent.answer_question(question, relevant_docs)
    print("\nAnswer:", answer)
