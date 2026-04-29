import re
import nltk
import math
from collections import Counter
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

def probability(tokens):
    # hitung frekuensi kata
    freq = Counter(tokens)
    # hitung total kata
    total = sum(freq.values())
    return {word: count / total for word, count in freq.items()}

def kl_divergence_flow(query, document):
    # preprocess query dan document
    q_tokens = preprocess(query)
    d_tokens = preprocess(document)
    
    # hitung probabilitas untuk query dan document
    p = probability(q_tokens)
    q = probability(d_tokens)
    
    # gabunngan vokab
    vocab = set(p.keys()) | set(q.keys())
    
    # smoothing dengan menambahkan epsilon untuk menghindari pembagian dengan nol
    epsilon = 1e-5
    kl_total = 0.0
    
    data = []
        
    # hitung KL Divergence
    for word in p.keys():
       p_word = p.get(word, epsilon)
       q_word = q.get(word, epsilon)
       kl_word = p_word * math.log(p_word / q_word)
       kl_total += kl_word
       
       if p_word > 0:
           data.append({
                "Word": word,
                "P": round(p_word, 6),
                "Q": round(q_word, 6),
                "KL": round(kl_word, 6)
           })
       
    return data, kl_total
    

def preprocess(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text.lower())
    tokens = [re.sub(r'[^\w\s]', '', t) for t in tokens] 
    tokens = [t for t in tokens if t]
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words and word.isalnum()]
    
    return filtered_tokens

if __name__ == "__main__":
    with open('artikel.md', 'r', encoding='utf-8') as file:
        content = file.read()
        
        # ambil judul artikel sebagai query
        title = ""
        for line in content.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        # hapus markdown syntax dari konten artikel
        clean_doc = re.sub(r'^\s*#{1,6}\s+.*$', '', content, flags=re.MULTILINE)
        
        print ("Query : ", title)
        print ("Document : ", clean_doc)
        
        data, result = kl_divergence_flow(title, clean_doc)
        print("KL Divergence:", round(result, 5))
        
    