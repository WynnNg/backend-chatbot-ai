from dotenv import load_dotenv
import os

from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

class Reflection():
    def __init__(self, vectorDB, embeddings, model_name="gpt-4o", type="openai"):
        self.model_name = model_name
        self.vectorDB = vectorDB
        self.embeddings = embeddings
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
    
    def __call__(self, chatHistory, originQuery ,lastItemsConsidered=50):
        if len(chatHistory) >= lastItemsConsidered:
            chatHistory = chatHistory[len(chatHistory)-lastItemsConsidered:]
        
        historyString = self._concat_and_format_texts(chatHistory)

        higherLevelSummariesPrompt = f"""
        Đây là lịch sử của cuộc trò chuyện: {historyString}.
        """

        originQueryVector = self.embeddings.encode([originQuery])
        synonymsQuestion = self.vectorDB.search(originQueryVector, "synonyms", 1)[0]
         
        questionStr = ''
        system_prompt = ''

        if synonymsQuestion['score'] >= 0.7 and 'synonyms_question' in synonymsQuestion['payload']:
            questionStr = synonymsQuestion['payload'].get('synonyms_question')
            
            print(f"check originQuery: {originQuery}")
            print(f"check synonymsQuestion: {questionStr}")

            system_prompt = f"""
            Bạn đóng vai khách hàng hãy tóm tắt đoạn lịch sử chat được cho bằng một câu hỏi đúng với ngữ cảnh mới nhất.
            - Nếu khách hỏi "{originQuery}" có nghĩa là "{questionStr}"
            """

        else:
            system_prompt = f"""
            Bạn đóng vai khách hàng hãy tóm tắt đoạn lịch sử chat được cho bằng một câu hỏi đúng với ngữ cảnh mới nhất.
            """

        if self.type == "openai":
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user", 
                        "content": higherLevelSummariesPrompt
                    },  {
                    "role": "system",
                    "content": system_prompt
                },
                ]
            )
            return completion.choices[0].message.content + questionStr
            
        else:
            raise NotImplementedError("Reflection for this type is not implemented.")
        return ""

            