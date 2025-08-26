from dotenv import load_dotenv
import os

from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_required
from flask_cors import CORS
from openai import OpenAI

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import desc
from db import db
from models import User, QA, Prompt

from vector_database import VectorDB

from embeddings import Embeddings

from rag.core import RAGChatBot
from reflection import Reflection

from semantic_router import SemanticRouter, Route
from semantic_router import carScreenSamples, androidBoxSamples, camera360Samples, clarifySamples, brandInfoSamples

from data_processor import DataProcessor

# Load environment variables from .env file
load_dotenv()

# Access the key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


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
    
# Initialize embeddings
openAIEmbedding = Embeddings(model_name="text-embedding-3-small", type="openai")

# Initialize OpenAI client
client_openai = OpenAI(api_key=OPENAI_API_KEY)

# Initialize vector database
vectorDB = VectorDB(db_type="qdrant")

# Initialize RAGChatBot with OpenAI client
rag_chatbot = RAGChatBot(client_openai, vectorDB)

# Initialize reflection
reflection = Reflection()

# Initialize DataProcessor
data_processor = DataProcessor(embedding=openAIEmbedding)

# --- SEMANTIC ROUTER Setup --- #
CAR_SCREEN_ROUTE_NAME = 'car_screen'
ANDROID_BOX_ROUTE_NAME = 'android_box'
CAMERA_360_ROUTE_NAME = 'camera_360'
CLARIFY_QUESTION_ROUTE_NAME = 'clarify_question'
BRAND_INFO_ROUTE_NAME = 'brand_info'

carScreenRoute = Route(name=CAR_SCREEN_ROUTE_NAME, samples=carScreenSamples)
androidBoxRoute = Route(name=ANDROID_BOX_ROUTE_NAME, samples=androidBoxSamples)
camera360Route = Route(name=CAMERA_360_ROUTE_NAME, samples=camera360Samples)
brandInfoRoute = Route(name=BRAND_INFO_ROUTE_NAME, samples=brandInfoSamples)
clarifyRoute = Route(name=CLARIFY_QUESTION_ROUTE_NAME, samples=clarifySamples)

semanticRouter = SemanticRouter(openAIEmbedding, routes=[carScreenRoute, androidBoxRoute, camera360Route, brandInfoRoute, clarifyRoute])
# --- End SEMANTIC ROUTER Setup --- #


# @login_manager.user_loader
def load_user(id):
    return User.query.get(int(id))

# @app.route('/')
# def home():
#     return "Welcome to the Flask App!"

def process_query(query):
    return query.lower()

# Route for chatbot interaction
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        # Get the query and chat history from the request
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({"error": "No query provided"}), 400
        
        query = data['query']
        chat_history = data.get('chat_history', '')

        query = process_query(query)

        # Use the semantic router to guide the query
        guidedRoute = semanticRouter.guide(query=query)[1]

        # Perform reflection on the chat history
        reflection_question = reflection(chat_history)

        # Get the system prompt from the database
        system_prompt = Prompt.query.filter(Prompt.prompt_id == 1).first()

        rag_completion = ""
        collection_name = ""

        # Perform RAG based on the guided route
        if guidedRoute == CAR_SCREEN_ROUTE_NAME:
            collection_name = "car_screen"
        elif guidedRoute == ANDROID_BOX_ROUTE_NAME:
            collection_name = "android_box"
        elif guidedRoute == CAMERA_360_ROUTE_NAME:
            collection_name = "camera_360"
        elif guidedRoute == BRAND_INFO_ROUTE_NAME:
            collection_name = "brand_info"
        elif guidedRoute == CLARIFY_QUESTION_ROUTE_NAME:
            collection_name = "clarify_question"
        
        print(f"sematic route: {guidedRoute}")

        rag_completion = rag_chatbot.perform_rag(reflection_question, collection_name, system_prompt, openAIEmbedding)
 
        if not rag_completion:
            return jsonify({"error": "No results found"}), 404

        return jsonify({"response": rag_completion}), 200

    except ValueError as e:
        return jsonify({"error": f"Invalid JSON: {str(e)}"}), 400

#Route for add documents to vector database
@app.route('/api/doc/learn', methods=['POST'])
def add_doc_to_db():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        response = data_processor.add_data(vectorDB, data, collection_name=data["collection_name"])
        
        if response.get("status") == "error":
            return jsonify(response), 500

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

#Route for update documents to vector database
@app.route('/api/doc/learn', methods=['PUT'])
def update_doc_to_db():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        response = data_processor.update_data(vectorDB, data, collection_name=data["collection_name"])
        
        if response.get("status") == "error":
            return jsonify(response), 500

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

#Route for delete documents to vector database
@app.route('/api/doc/delete', methods=['DELETE'])
def delete_doc_to_db():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        response = vectorDB.delete_item(collection_name=data["collection_name"], item_id=data["id"])
        
        if response.get("status") == "error":
            return jsonify(response), 500

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

#Route for delete collection to vector database
@app.route('/api/collection/delete', methods=['DELETE'])
def delete_collection_to_db():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        response = vectorDB.delete_collection(collection_name=data["collection_name"])
        
        if response.get("status") == "error":
            return jsonify(response), 500

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/system-prompt', methods=['POST'])
def add_system_prompt():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        print("check data", data)
        prompt = Prompt.query.filter(Prompt.prompt_id == 1).first()
        print("check prompt", prompt)

        if prompt:
            prompt.prompt = data["prompt"]
        else:
            prompt = Prompt(prompt_id = 1, prompt=data["prompt"])

        db.session.add(prompt)

        try:
            db.session.commit()
        except:
            db.session.rollback()

        response = {
            "prompt_id": prompt.prompt_id,
            "prompt": prompt.prompt
        }

        return jsonify({"data": response, "status": "success"}), 200

    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == '__main__':
    app.run()