from rerank import Rerank

class RAGChatBot():
    def __init__(self, llm, vectorDB):
        self.llm = llm
        self.vectorDB = vectorDB
        self.reranker = Rerank()

    def get_completion(self, prompt, system_prompt):

        if system_prompt:
            sys_prompt = system_prompt.prompt
        else:
            sys_prompt = """ 
                Bạn tên là An, là một nhân viên trả lời và chăm sóc khách hàng chuyên nghiệp của thương hiệu DCTECH, có nhiệm vụ tư vấn và trả lời khách hàng về các sản phẩm và dịch vụ liên quan đến: Màn hình giải trí cao cấp, camera 360, android box cho ô tô.
                Bạn là một nhân viên lễ phép luôn bắt đầu từ "Dạ" và kết thúc từ "ạ" trong mỗi câu tư vấn.
                Bạn xưng mình là "em" và gọi khách hàng là "anh/chị".
                Bạn Luôn tuân thủ đúng vai trò, không nói vượt ngoài phạm vi cho phép.
                Mọi thông tin cần tra cứu tại:
                DCTECH - Drive The Future
                Hotline: 0339 1111 88 
                Website: https://dctechauto.com/
            """ 

        response = self.llm.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": sys_prompt
                },
                {"role": "user", "content": prompt},
            ]
        )
        return response.choices[0].message.content
    
    def enhance_prompt(self, query, query_vector, collection_name, limit):
        knowledges = self.vectorDB.search(query_vector, collection_name, limit)
        
        i = 0
        docs = []
        for result in knowledges:
            if result['payload'] is not None:
                result_str = ""
                if result['payload'].get('name'):
                    result_str += f"\n - Tên: {result['payload'].get('name')}"
                if result['payload'].get('price'):
                    result_str += f", giá: {result['payload'].get('price')}"
                if result['payload'].get('promotion'):
                    result_str += f", khuyến mãi: {result['payload'].get('promotion')}"
                if result['payload'].get('product_info'):
                    result_str += f", Thông tin sản phẩm: {result['payload'].get('product_info')}"
                if result['payload'].get('content'):
                    result_str += f", Thông tin: {result['payload'].get('content')}"
                docs.append(result_str)

        reranked_docs = self.reranker.rerank_docs(query, docs)
        reranked_str = "\n".join(reranked_docs)

        return reranked_str if reranked_str else "Không có thông tin liên quan"

    def make_rag_prompt(self, query, results):
        return f"""<INSTRUCTIONS>
        You are an expert sales and consultation specialist for DCTECH. 
        Your task is to answer the following user question. The search results of a document search have been included to give you more context. 
        Use the information in Search Results to help you answer the question accurately.
        Not all information in the search results will be useful. 
        However, if you find any information that's useful for answering the user's question, draw from it in your answer.
        </INSTRUCTIONS>

        <USER QUERY>
        {query}
        </USER QUERY>

        <SEARCH RESULTS>
        {results}
        </SEARCH RESULTS>
        Your answer:"""
    
    def perform_rag(self, query, collection_name, system_prompt, embedding, limit=5):
        query_vector = embedding.encode([query])

        # Get knownledges from database
        result_str = self.enhance_prompt(query, query_vector, collection_name, limit)

        # Create the RAG prompt
        rag_prompt = self.make_rag_prompt(query, result_str)

        # Get the RAG completion
        rag_completion = self.get_completion(rag_prompt, system_prompt)

        return rag_completion