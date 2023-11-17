#text generation
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
output = generator("I want to buy some chocolate, should I go to the shop or should I not?", max_length=30, num_return_sequences=5)

print (output)

#embeddings
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2',cache_folder='VectorizationModels')
embedddings = model.encode(["I want to buy some chocolate, should I go to the shop or should I not?"])