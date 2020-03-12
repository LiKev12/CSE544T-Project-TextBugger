import tensorflow as tf
import tensorflow_hub as hub

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4" #@param ["https://tfhub.dev/google/universal-sentence-encoder/4", "https://tfhub.dev/google/universal-sentence-encoder-large/5"]
model = hub.load(module_url)
print ("module %s loaded" % module_url)
def embed(input):
  return model(input)

def semantic_similarity(tokens1, tokens2):
    sts_encode1 = tf.nn.l2_normalize(embed([" ".join(tokens1)]), axis=1)
    sts_encode2 = tf.nn.l2_normalize(embed([" ".join(tokens2)]), axis=1)
    cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
    return cosine_similarities



# res = semantic_similarity(
#     ['I','love','hugging','joyful','people'],
#     ['I','love','hurting','joyful','people']
# )

# print(res)