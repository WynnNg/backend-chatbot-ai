
class RAGChatBot():
    def __init__(self, llm, dbCollection, sys_prompt=""):
        self.llm = llm
        self.collection = dbCollection
        self.sys_prompt = sys_prompt
    
    def get_completion(self, prompt):

        if self.sys_prompt:
            system_prompt = self.sys_prompt
        else:
            system_prompt = """ 
                Bạn tên là An, là một nhân viên trả lời và chăm sóc khách hàng chuyên nghiệp của thương hiệu DCTECH, có nhiệm vụ tư vấn và trả lời khách hàng về các sản phẩm và dịch vụ liên quan đến: Màn hình giải trí cao cấp, camera 360, android box cho ô tô.
                Bạn là một nhân viên lễ phép luôn bắt đầu từ "Dạ" và kết thúc từ "ạ" trong mỗi câu tư vấn.
                Bạn xưng mình là "em" và gọi khách hàng là "anh/chị".
                Bạn Luôn tuân thủ đúng vai trò, không nói vượt ngoài phạm vi cho phép.
                Mục tiêu:
                    - Hiểu đúng ý định và chủ đề của khách hàng để trả lời chính xác.
                    - Nếu khách hỏi chưa rõ, cần chủ động đặt 1–3 câu hỏi khai thác thêm.
                    - Giao tiếp tự nhiên, thân thiện, chuyên nghiệp và rõ ràng.

                Quy tắc:
                    1. Luôn phản hồi nhanh, lịch sự, ưu tiên hiểu đúng mục đích của khách.
                    2. Nếu khách chỉ nói mơ hồ như: “Tư vấn”, “Ib”, “Báo giá” → không trả lời ngay, mà đặt thêm 1 câu hỏi để xác định rõ khách cần gì.
                    3. Nếu khách đã cung cấp đủ “ý định” (intent) và “chủ đề” (topic) → tìm trong thư viện câu trả lời đã được nạp sẵn để phản hồi đúng.
                    4. Nếu không tìm thấy thông tin chính xác → giữ lại câu hỏi và xin phép hỗ trợ sau, không đoán bừa.

                Định nghĩa:
                    - "Ý định" là mục đích chính của khách như: hỏi giá, tư vấn sản phẩm, hỏi bảo hành, hỏi kỹ thuật, khiếu nại, v.v.
                    - "Chủ đề" là mã sản phẩm hoặc loại sản phẩm mà khách quan tâm như: DC H500, màn hình 2K, màn hình HD, Camera, Box Android, v.v.

                Câu trả lời cần:
                    - Lịch sự, tối ưu ngắn gọn
                    - Gợi mở nếu chưa đủ dữ kiện
                    - Tránh trả lời sai hoặc phán đoán không chắc chắn

                Ví dụ:
                    Nếu khách nhắn: “Mình muốn báo giá”, AI nên hỏi lại:  
                        “Dạ anh/chị cần báo giá sản phẩm nào ạ? Hiện em đang hỗ trợ các dòng như TPMS, Camera hành trình, Màn hình Android…”

                Mọi thông tin cần tra cứu tại:
                DCTECH - Drive The Future
                Hotline: 0339 1111 88 
                Website: https://dctechauto.com/
            """ 

        response = self.llm.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message.content

    def populate_rag_query(self, query, n_results=1):
        search_results = self.collection.query(query_texts=[query], n_results=n_results)
        result_str = ""
        for idx, result in enumerate(search_results["documents"][0]):
            metadata = search_results["metadatas"][0][idx]
            formatted_result = f"""<SEARCH RESULT>
            <DOCUMENT>{result}</DOCUMENT>
            <METADATA>
            <TITLE>{metadata['title']}</TITLE>
            <QUESTION>{metadata['question']}</QUESTION>
            <ANSWER>{metadata['answer']}</ANSWER>
            <CHUNK_IDX>{metadata['id']}</CHUNK_IDX>
            <TOPIC>{metadata['topic']}</TOPIC>
            </METADATA>
            </SEARCH RESULT>"""
            result_str += formatted_result
        return result_str

    def make_rag_prompt(self, query, results):
        return f"""<INSTRUCTIONS>
        Your task is to answer the following user question. The search results of a document search have been included to give you more context. Use the information in Search Results to help you answer the question accurately.
        Not all information in the search results will be useful. However, if you find any information that's useful for answering the user's question, draw from it in your answer.
        </INSTRUCTIONS>

        <USER QUERY>
        {query}
        </USER QUERY>

        <SEARCH RESULTS>
        {results}
        </SEARCH RESULTS>
        Your answer:"""
    
    def rewrite_query(self, query, chat_history):
        prompt = f"""<INSTRUCTIONS>
            Given the following chat history and the user's latest query, rewrite the query to include relevant context.
            </INSTRUCTIONS>
            <CHAT_HISTORY>
            {chat_history}
            </CHAT_HISTORY>
            <LATEST_QUERY>
            {query}
            </LATEST_QUERY>
            Your rewritten query:"""
        return self.get_completion(prompt)
    
    def perform_cqr_rag(self, query, chat_history, n_results=1):
        # Rewrite the query using the chat history
        refined_query = self.rewrite_query(query, chat_history)
        result_str = self.populate_rag_query(refined_query, n_results)
        rag_prompt = self.make_rag_prompt(refined_query, result_str)
        rag_completion = self.get_completion(rag_prompt)
        return rag_completion