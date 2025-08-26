from dotenv import load_dotenv
import os

from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

class Reflection():
    def __init__(self, model_name="gpt-4o", type="openai"):
        self.model_name = model_name
        self.type = type

        if self.type == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def _concat_and_format_texts(self, data):
        concatenatedTexts = []
        for item in data:
            role = item.get("role")
            content = item.get("content")
            if role and content:
                concatenatedTexts.append(f"{role}: {content} \n")
        return "".join(concatenatedTexts)
    
    def __call__(self, chatHistory, lastItemsConsidered=100):
        if len(chatHistory) >= lastItemsConsidered:
            chatHistory = chatHistory[len(chatHistory)-lastItemsConsidered:]
        
        historyString = self._concat_and_format_texts(chatHistory)

        higherLevelSummariesPrompt = f"""
        Given a chat history and the latest user question which might reference context in the chat history, 
        formulate a standalone question in Vietnamese which can be understood without the chat history. 
        Do NOT answer the question, just reformulate it if needed and otherwise return it as is. {historyString}
        """

        if self.type == "openai":
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user", 
                        "content": higherLevelSummariesPrompt
                    }
                ]
            )
            return completion.choices[0].message.content
            
        else:
            raise NotImplementedError("Reflection for this type is not implemented.")
        return ""

            