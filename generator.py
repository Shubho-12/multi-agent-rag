# generator.py
from transformers import pipeline

class TextGenerator:
    def __init__(self, device=-1):
        """
        device=-1 means CPU, >=0 means GPU device id
        """
        print("Loading generation model (t5-small) on CPU...")
        self.generator = pipeline(
            "text2text-generation",
            model="t5-small",
            tokenizer="t5-small",
            device=device  # -1 for CPU
        )
        print("Model loaded.")

    def generate(self, prompt, max_length=150):
        """
        Generate text from a prompt using the model
        """
        result = self.generator(prompt, max_length=max_length)
        return result[0]['generated_text'] if result else ""
