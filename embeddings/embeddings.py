import os
from openai import OpenAI
from google import genai
from dotenv import load_dotenv

load_dotenv()

class Embeddings:
    def __init__(self, model_name, type):
        self.model_name = model_name
        self.type = type
        if self.type == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.type == "gemini":
            self.client = genai.Client(
                api_key=os.getenv("GEMINI_API_KEY"),
            )
    
    def encode(self, doc):
        if self.type == "openai":
            return self.client.embeddings.create(
                model=self.model_name,
                input=doc,
                dimensions=768
            ).data[0].embedding
        elif self.type == "gemini":
            response = self.client.models.embed_content(
                model=self.model_name,
                content=doc
            )
            return response.embeddings[0].values
        return []
