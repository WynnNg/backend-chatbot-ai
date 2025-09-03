import numpy as np

class SemanticRouter():
    def __init__(self, embedding, routes):
        self.routes = routes
        self.embedding = embedding
        self.route_embeddings = {}
        # self.sample_embeddings = {}

        for route in self.routes:
            self.route_embeddings[route.name] = self.embedding.encode(route.samples)
            # samplesEmbedding = []
            # for sample in route.samples:
            #     sampleEmbedding = self.embedding.encode(sample)
            #     samplesEmbedding.append(sampleEmbedding)
            # self.sample_embeddings[route.name] = samplesEmbedding

    # def is_cosine_similarity(self, queryEmbedding, routeName, threshold=0.7):
    #     samplesEmbedding = self.sample_embeddings[routeName]
    #     scores = []

    #     for i, sample in enumerate(samplesEmbedding):
    #         samplesEmbedding[i] = samplesEmbedding[i]/np.linalg.norm(samplesEmbedding[i])
    #         score = np.mean(np.dot(samplesEmbedding[i], queryEmbedding.T).flatten())
    #         scores.append(score)

    #     # Sắp xếp scores theo thứ tự giảm dần
    #     scores.sort(reverse=True)

    #     # Trả về cả scores và kết quả kiểm tra
    #     return scores[0] >= threshold

            
    def guide(self, query):
        queryEmbedding = self.embedding.encode([query])
        queryEmbedding = queryEmbedding/np.linalg.norm(queryEmbedding)
        scores = []

        # calculate the cosine similarity of the query embedding with the sample embeddings of the router.
        for route in self.routes:
            # if self.is_cosine_similarity(queryEmbedding, route.name):
            #     return ('', route.name)    
            routeEmbedding = self.route_embeddings[route.name]
            routeEmbedding = routeEmbedding / np.linalg.norm(routeEmbedding)
            score = np.mean(np.dot(routeEmbedding, queryEmbedding.T).flatten())
            scores.append((score, route.name))

        scores.sort(reverse=True)
        return scores[0]





