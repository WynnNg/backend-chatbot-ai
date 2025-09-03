class DataProcessor():
    def __init__(self, embedding):
        self.embedding = embedding

    def chunk_text(self, text, chunk_size=500, overlap=50):
        pass
    
    def add_data(self, vector_db , data, collection_name):

        if not data['id']:
            raise ValueError("Data must contain 'id' fields.")

        doc_texts = ""
        payload = {k: v for k, v in data.items()}

        if 'name' in data:
            doc_texts += f"Tên: {data['name']}. "
        if 'price' in data:
            doc_texts += f"Giá: {data['price']}. "
        if 'promotion' in data:
            doc_texts += f"Khuyến mãi: {data['promotion']}. "
        if 'product_info' in data:
            doc_texts += f"Thông tin sản phẩm: {data['product_info']}. "
        if 'content' in data:
            doc_texts += f"Nội dung: {data['content']}. "
        if 'origin_question' in data:
            doc_texts += f"{data['origin_question']}. "
                  
        doc_vector = self.embedding.encode([doc_texts])

        data_to_save = {
            "id": data['id'],
            "vector": doc_vector,
            "payload": payload
        }

        res = vector_db.add_item(data_to_save, collection_name)

        return res
    
    def update_data(self, vector_db, data, collection_name):
        if not data['id']:
            raise ValueError("Data must contain 'id' fields.")

        doc_texts = ""
        payload = {k: v for k, v in data.items()}
        
        if 'name' in data:
            doc_texts += f"Tên: {data['name']}. "
        if 'price' in data:
            doc_texts += f"Giá: {data['price']}. "
        if 'promotion' in data:
            doc_texts += f"Khuyến mãi: {data['promotion']}. "
        if 'product_info' in data:
            doc_texts += f"Thông tin sản phẩm: {data['product_info']}. "
        if 'content' in data:
            doc_texts += f"Nội dung: {data['content']}. "
        if 'origin_question' in data:
            doc_texts += f"{data['origin_question']}. "

        doc_vector = self.embedding.encode([doc_texts])

        data_to_update = {
            "id": data['id'],
            "vector": doc_vector,
            "payload": payload
        }

        res = vector_db.update_item(data_to_update, collection_name)
        return res
        
        