import numpy as np

class SemanticRouter():
    def __init__(self, embedding, routes):
        self.routes = routes
        self.embedding = embedding
        self.route_embeddings = {}

        for route in self.routes:
            self.route_embeddings[route.name] = self.embedding.encode(route.samples)
            
    def guide(self, query):
        queryEmbedding = self.embedding.encode([query])
        queryEmbedding = queryEmbedding/np.linalg.norm(queryEmbedding)
        scores = []

        # calculate the cosine similarity of the query embedding with the sample embeddings of the router.
        for route in self.routes:
            routeEmbedding = self.route_embeddings[route.name]
            routeEmbedding = routeEmbedding / np.linalg.norm(routeEmbedding)
            score = np.mean(np.dot(routeEmbedding, queryEmbedding.T).flatten())
            scores.append((score, route.name))

        scores.sort(reverse=True)
        return scores[0]





