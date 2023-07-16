import tensorflow_hub as hub
model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
nlp_module = hub.load(model_url)
#########################################################################
#########################################################################
#TensorHub
#Installation
import tensorflow_hub as hub

#Browse Models
 # Vision Model
 model_url = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/2"
 vision_module = hub.Module(model_url)
 # NLP Model
 model_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
 nlp_module = hub.load(model_url)
 # Audio Model
 model_url = "https://tfhub.dev/google/nonsemantic-speech-benchmark/trill/3"
 audio_module = hub.Module(model_url)

#Example Pretrained model
sentence = "The quick brown fox jumps over the lazy dog"
vector = nlp_module([sentence])[0]
vector



