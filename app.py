from dotenv import load_dotenv
import os

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_required
from flask_cors import CORS
from openai import OpenAI

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import desc
import chromadb
from db import db

from models import User, QA
from embeddings import ChromaDB

from rag.core import RAGChatBot

# Load environment variables from .env file
load_dotenv()
# Access the key
OPEN_AI_KEY = os.getenv('OPEN_AI_KEY')

app = Flask(__name__)
CORS(app)
app.config['SECRET_KEY'] = 'mysecretkey'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///myDB.db' #path to database and its name
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False #to supress warning


# login_manager = LoginManager()
# login_manager.init_app(app)
db.init_app(app) #database instance
with app.app_context():
    db.create_all()  # Create database tables

client_openai = OpenAI(api_key=OPEN_AI_KEY)

# Đường dẫn tới thư mục chroma database
db_name = "chromaDatabase"
# Kiểm tra và tạo thư mục nếu chưa tồn tại
if not os.path.exists(f"./{db_name}"):
    os.makedirs(f"./{db_name}")
# Khởi tạo client với đường dẫn
client_chroma = chromadb.PersistentClient(path=f"./{db_name}")
collection = client_chroma.get_or_create_collection(name=db_name, metadata={"hnsw:space": "cosine"})

# Initialize RAGChatBot with OpenAI client and ChromaDB collection
rag_chatbot = RAGChatBot(client_openai, collection)
chromadb = ChromaDB(client_chroma, collection)

# @login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))

# @app.route('/')
# def home():
#     return "Welcome to the Flask App!"

# Route for chatbot interaction
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        print(data)
        if not data or 'query' not in data:
            return jsonify({"error": "No query provided"}), 400

        query = data['query']
        chat_history = data.get('chat_history', '')

        rag_completion = rag_chatbot.perform_cqr_rag(query, chat_history, n_results=2)
        
        if not rag_completion:
            return jsonify({"error": "No results found"}), 404

        return jsonify({"response": rag_completion}), 200

    except ValueError as e:
        return jsonify({"error": f"Invalid JSON: {str(e)}"}), 400    

#Route for question and answer management
@app.route('/api/qa')
def get_all_qa():
    try:
        qa_list = QA.query.order_by(desc(QA.created_at)).all()
        if not qa_list:
            return jsonify({"data": []}), 200

        result = []
        for qa in qa_list:
            result.append({
                "id": qa.id,
                "topic": qa.topic,
                "question": qa.question,
                "answer": qa.answer,
                "content": qa.content,
                "is_chunk": qa.is_chunk
            })

        return jsonify({"data": result}), 200

    except SQLAlchemyError as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/qa/<int:qa_id>', methods=['GET'])
def get_qa_by_id(qa_id):
    try:
        qa = QA.query.get(qa_id)
        if not qa:
            return jsonify({"error": "QA not found"}), 404

        return jsonify({
            "data": {
                "id": qa.id,
                "topic": qa.topic,
                "question": qa.question,
                "answer": qa.answer,
                "content": qa.content,
                "is_chunk": qa.is_chunk
            }
        }), 200

    except SQLAlchemyError as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route('/api/add/qa', methods=['POST'])
def add_qa():
    try:
        data = request.get_json()
      
        if not data:
            return jsonify({"error": "No data provided"}), 400

        required_fields = ['question', 'answer']
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields: 'question' or 'answer'"}), 400

        new_qa = QA(
            topic=data.get('topic', ''),
            question=data['question'],
            answer=data['answer'],
            content= f"Chủ đề: {data.get('topic', '')}. Câu hỏi: {data['question']}. Câu Trả lời {data['answer']}", 
        )

        db.session.add(new_qa)
        try:
            db.session.commit()
        except:
            db.session.rollback()

        return jsonify({"data": {
                "id": new_qa.id,
                "topic": new_qa.topic,
                "question": new_qa.question,
                "answer": new_qa.answer,
                "content": new_qa.content,
                "is_chunk": new_qa.is_chunk
            }}), 201

    except ValueError as e:
        return jsonify({"error": f"Invalid JSON: {str(e)}"}), 400
    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/delete/qa/<int:qa_id>', methods=['DELETE'])
def delete_qa(qa_id):
    try:
        qa_to_delete = QA.query.get(qa_id)
        if not qa_to_delete:
            return jsonify({"error": "QA not found"}), 404
        
        if qa_to_delete.is_chunk:
            metadata = {
                    "id": qa_to_delete.id,
                    "title": qa_to_delete.question,
                    "question": qa_to_delete.question,
                    "answer": qa_to_delete.answer,
                    "topic": qa_to_delete.topic
            }
            chromadb.delete_document(
                metadatas=metadata
            )

        db.session.delete(qa_to_delete)
        db.session.commit()

        return jsonify({"message": "QA deleted successfully"}), 200

    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/update/qa/<int:qa_id>', methods=['PUT'])
def update_qa(qa_id):
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        qa_to_update = QA.query.get(qa_id)
        if not qa_to_update:
            return jsonify({"error": "QA not found"}), 404

        if 'question' in data:
            qa_to_update.question = data['question']
        if 'answer' in data:
            qa_to_update.answer = data['answer']
        if 'topic' in data:
            qa_to_update.topic = data['topic']
        if 'is_chunk' in data:
            qa_to_update.is_chunk = data['is_chunk']
            metadata = {
                "id": qa_to_update.id,
                "title": qa_to_update.question,
                "question": qa_to_update.question,
                "answer": qa_to_update.answer,
                "topic": qa_to_update.topic
            }
            chromadb.add_documents(
                documents=qa_to_update.content,
                metadatas=metadata
            )

        db.session.commit()

        return jsonify({"data": {
                "id": qa_to_update.id,
                "topic": qa_to_update.topic,
                "question": qa_to_update.question,
                "answer": qa_to_update.answer,
                "content": qa_to_update.content,
                "is_chunk": qa_to_update.is_chunk
            }}), 200

    except SQLAlchemyError as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8501, debug=True)