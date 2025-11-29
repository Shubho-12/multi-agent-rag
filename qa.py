# qa.py
from generator import TextGenerator

class QAgent:
    def __init__(self):
        # CPU only
        self.generator = TextGenerator(device=-1)

    def answer_question(self, question, docs):
        """
        Combines all retrieved docs as context and generates an answer
        """
        context = " ".join(docs)
        prompt = f"Answer the question based on the context:\nContext: {context}\nQuestion: {question}"
        answer = self.generator.generate(prompt)
        return answer
