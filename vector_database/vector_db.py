from qdrant_client import QdrantClient
from supabase import create_client, Client
from dotenv import load_dotenv
from qdrant_client import models as qdrant_models
import os

load_dotenv()

class VectorDB:
    def __init__(self, db_type: str):
        self.db_type = db_type

        if self.db_type == "qdrant":
            self.client = QdrantClient(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY")
            )
            self.ping()
        elif self.db_type == "supabase":
            url: str = os.getenv("SUPABASE_URL")
            key: str = os.getenv("SUPABASE_KEY")
            supabase: Client = create_client( 
                supabase_url=url, 
                supabase_key=key
            )
            self.client = supabase
            self.ping()
    
    def add_item(self, data, collection_name):
        #--- Add an item to the vector database ---
        if self.db_type == "qdrant":
            # Prepare data for Qdrant
            id = data.get("id")
            vector = data.get("vector")
            payload = data.get("payload", {})
            
            if not id or not vector:
                raise ValueError("Data must contain 'id' and 'vector' fields.")
            try:
                collections = self.client.get_collections().collections
                collection_names = [col.name for col in collections]

                if collection_name not in collection_names:
                    # Create collection if it doesn't exist
                    try:
                        self.client.create_collection(
                            collection_name=collection_name,
                            vectors_config=qdrant_models.VectorParams(
                                size=len(vector),
                                distance=qdrant_models.Distance.COSINE
                            )
                        )
                        print(f"Collection {collection_name} created successfully in Qdrant.")
                    
                    except Exception as e:
                        print(f"Failed to create collection {collection_name}: {str(e)}")
                
                # Add item to the collection
                result = self.client.upsert(
                    collection_name=collection_name,
                    points=[
                        qdrant_models.PointStruct(
                            id=id,
                            vector=vector,
                            payload=payload
                        )
                    ]                
                )
                return {"status": "success", "operation": "upsert", "collection": collection_name, "result": f"{result}"}
            except Exception as e:
                return {"status": "error", "operation": "upsert", "collection": collection_name, "error": str(e)}
        else:
            print(f"Unsupported database: {self.db_type}")
    
    def update_item(self, data, collection_name):
        if self.db_type == "qdrant":
            id = data.get("id")
            vector = data.get("vector")
            payload = data.get("payload", {})
            if not id or not vector or not payload:
                raise ValueError("Data must contain 'id', 'vector' and 'payload' fields.")
            try:

                # Update vector on the given points
                self.client.update_vectors(
                    collection_name=collection_name,
                    points=[
                        qdrant_models.PointVectors(
                            id=id,
                            vector=vector,
                        )
                    ]
                )

                # Update payload if provided
                self.client.set_payload(
                    collection_name=collection_name,
                    payload=payload,
                    points=[id],
                )

                return {"status": "success", "operation": "update_item", "collection": collection_name, "item_id": f"{id}"}
            except Exception as e:
                return {"status": "error", "operation": "update_item", "collection": collection_name, "item_id": f"{id}", "error": str(e)}

        
    def search(self, query_vector, collection_name, limit=5):
        if self.db_type == "qdrant":
            if self.client.collection_exists(collection_name=collection_name):
                search_result = self.client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    limit=limit
                ).points

                # Extract the results
                results = []
                for point in search_result:
                    results.append({
                        "id": point.id,
                        "score": point.score,
                        "payload": point.payload
                    })

                # Sort by score to get highest scores first
                results.sort(key=lambda x: x.get('score', 0), reverse=True)
                return results
            else:
                return []
        else:
            print(f"Unsupported database: {self.db_type}")
            return []
    
    def delete_item(self, collection_name, item_id):
        if self.db_type == "qdrant":
            try:
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=qdrant_models.PointIdsList(
                        points=[item_id]
                    )
                )
                return {"status": "success", "operation": "delete_item", "collection": collection_name, "item_id": str(item_id)}
            except Exception as e:
                return {"status": "error", "operation": "delete_item", "collection": collection_name, "item_id": str(item_id), "error": str(e)}
        else:
            print(f"Unsupported database: {self.db_type}")
            return {"status": "error", "operation": "delete_item", "collection": collection_name, "item_id": str(item_id), "error": "Unsupported database type"}
    
    def delete_collection(self, collection_name):
        if self.db_type == "qdrant":
            if client.collection_exists(collection_name=collection_name):
                try:
                    self.client.delete_collection(collection_name=collection_name)
                    return {"status": "success", "operation": "delete_collection", "collection": collection_name}
                except Exception as e:
                    return {"status": "error", "operation": "delete_collection", "collection": collection_name, "error": str(e)}
            else:
                return {"status": "success", "operation": "delete_collection", "collection": f"{collection_name} doesn't exist!"}
        else:
            print(f"Unsupported database: {self.db_type}")
            return {"status": "error", "operation": "delete_collection", "collection": collection_name, "error": "Unsupported database type"}

    def ping(self):
        """ Check if the database is alive and reachable """

        try:
            if self.db_type == "qdrant":
                try:
                    # Get the list of collections
                    collections = self.client.get_collections()
                    # If we get here without an exception, the connection is successful
                    print(f"Qdrant connection successful. Found {len(collections.collections)} collections.")
                    return True, f"Qdrant connection successful. Found {len(collections.collections)} collections."
                except Exception as e:
                    # If we can't get collections, try a simpler healthcheck
                    try:
                        cluster_info = self.client.cluster_info()
                        print(f"Qdrant connection successful. (Verified via cluster_info)")
                        return True, "Qdrant connection successful."
                    except Exception as e2:
                        print(f"Qdrant connection failed: {str(e2)}")
                        return False, f"Qdrant connection failed: {str(e2)}"

            elif self.db_type == "supabase":
                # For Supabase 2.x
                try:
                    # Intentionally query a non-existent table
                    # This will fail, but the type of error tells us if connection works
                    response = self.client.from_("nonexistent_table_for_health_check").select("*").limit(1).execute()
                    # If we somehow get here without an exception, the connection is successful
                    print("Supabase connection successful.")
                    return True, "Supabase connection successful."
                except Exception as e:
                    # Check the error message
                    error_msg = str(e)
                    # If we get a "Table doesn't exist" error, the connection is likely fine"
                    if "42P01" in error_msg or "does not exist" in error_msg:
                        print("Supabase connection successful. (verified via error message)")
                        return True, "Supabase connection successful. (verified via error message)"
                    else:
                        print(f"Supabase connection failed: {error_msg}")
                        return False, f"Supabase connection failed: {error_msg}"

            else:
                print(f"Unsupported database type: {self.db_type}")

        except Exception as e:
            print(f"{self.db_type} connection failed: {str(e)}")
