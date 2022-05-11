import sys
import spacy
import numpy as np
import fasttext
from joblib import dump, load
import statistics
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy.symbols import ORTH
from luima_sbd import sbd_utils
from luima_sbd.sbd_utils import text2sentences
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

#corpus = sys.argv[1] # 0th element is 'analyze.py' name
#print("Running....")

def spacy_tokenize(txt):
    doc = nlp(txt)
    tokens = list(doc)
    clean_tokens = []
    for t in tokens:
        if t.pos_ == 'PUNCT':
            pass
        elif t.pos_ == 'NUM':
            clean_tokens.append(f'<NUM{len(t)}>')
        else:
            clean_tokens.append(t.lemma_)
    return clean_tokens


with open(sys.argv[1], 'r') as f:
    contents = f.read()
f.close()

#print("Featurizing Text...")
# Producing a list of sentences 
list_of_sentences = list()
for sentence in list(text2sentences(contents)):
    list_of_sentences.append(sentence)


# Retrieving Offsets of sentences
sentence_offsets = list()
for sentence in list(text2sentences(contents, offsets=True)):
    sentence_offsets.append(sentence)
    
# Generating normalized sentence start list
normalized_sentence_start = list()
for i in range(len(list_of_sentences)):
    normalized_sentence_start.append(sentence_offsets[i][0]/((float)(len(contents))))
    

# Retrieveing number of tokens per each sentence
nlp = spacy.load("en_core_web_sm")
sentence_token_number = list()
for i in range(len(list_of_sentences)):
    doc = nlp(list_of_sentences[i])
    sentence_token_number.append(len(list(doc)))


# Standard deviation of number of tokens in entire document
std_dev = statistics.pstdev(sentence_token_number)
# Average number of tokens per sentence in document
average_sentence_tokens = sum(sentence_token_number)/(float)(len(sentence_token_number))

# Normalized number of tokens per each sentence:
normalized_tokens_per_sentence = list()
for i in range(len(list_of_sentences)):
    doc = nlp(list_of_sentences[i])
    normalized_tokens_per_sentence.append((len(list(doc))-average_sentence_tokens)/std_dev)
    
# Loading learnt Embedding model
model = fasttext.load_model("model_luima.bin")
    
def generate_sentence_embedding(sentence):
    doc = nlp(sentence)
    embedding_vector = np.zeros([100,])
    for word in list(doc):
        embedding_vector += np.nan_to_num(model.get_word_vector(word.text))
    return np.nan_to_num(embedding_vector/len(list(doc)))

list_sentence_embeddings = np.empty([len(list_of_sentences), 100])
for i in range(len(list_of_sentences)):
    list_sentence_embeddings[i] = generate_sentence_embedding(list_of_sentences[i])

# Concatenating Embedding+Normalized Start+Normalized Tokens per Sentence
normalized_sentence_start = np.array(normalized_sentence_start)
normalized_tokens_per_sentence = np.array(normalized_tokens_per_sentence)
X_features = np.concatenate((list_sentence_embeddings, np.expand_dims(normalized_sentence_start, axis=1), np.expand_dims(normalized_tokens_per_sentence, axis=1)), axis=1)
#print("Running Model...")
# Loading and running model
model_exp = load('best2_model.joblib')
predictions = model_exp.predict(X_features)
#Outputting results
for i in range(len(predictions)):
    print(list_of_sentences[i])
    print("Predicted Type: ",predictions[i])
    print("\n--------------------------------------------------\n")
