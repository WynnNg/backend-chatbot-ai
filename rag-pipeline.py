from semantic_router import SemanticRouter, Route
from semantic_router import carScreenPriceSamples, carScreenInfoSamples, carScreenTechnicalSamples, androidBoxPriceSamples, androidBoxInfoSamples, androidBoxTechnicalSamples, ClarifySamples
from embeddings import Embeddings

carScreenPriceRoute = Route(
    name="carScreenPriceRoute",
    samples=carScreenPriceSamples
)

carScreenInfoRoute = Route(
    name="carScreenInfoRoute",
    samples=carScreenInfoSamples
)

router = SemanticRouter(
    embedding=Embeddings("text-embedding-3-small", "openai"),  # Replace with your actual embedding model 
    routes=[
        carScreenPriceRoute,
        carScreenInfoRoute,
    ]
)

print(router.guide(query="xin chào"))
