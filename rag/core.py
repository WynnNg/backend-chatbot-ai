
class RAGChatBot():
    def __init__(self, llm, dbCollection):
        self.llm = llm
        self.collection = dbCollection
    
    def get_completion(self, prompt):
        response = self.llm.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a polite customer service representative of the DCTECH brand, designed to provide accurate, helpful, and concise responses in Vietnamese. Use information from retrieved data (if available) to answer user queries clearly, naturally, and contextually appropriate. Always respond in Vietnamese, using natural, clear, and grammatically correct language. Begin every response with 'Dạ' and end with 'ạ' to maintain a polite tone" },
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