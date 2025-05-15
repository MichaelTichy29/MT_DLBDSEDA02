import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
import re

from gensim.corpora.dictionary import Dictionary
from gensim import models
import warnings
warnings.filterwarnings("ignore")




#################################################
###########  Claening     #############
#################################################

###lowercase
### stopwords
# nltk

def cleaning_sub(df_test):
    stop = stopwords.words('english')
    
    df_test['narrative_str'] = df_test['narrative'].apply(str)
    
    #print(df_test['narrative_str'])
    #print(type(df_test['narrative_str']))
    
    def remove_stopwords(text):
        text = re.sub(r'[^\w\s]','',text).lower()
        text = ' '.join([word for word in text.split() if word not in stop])
        return text
    
    df_test['narrative_ws'] = df_test['narrative_str'].apply(remove_stopwords)
    #print(df_test['Text_ws'])
    
    df_test['narrative_ws'] = \
    df_test['narrative_ws'].map(lambda x: re.sub('[,\.!?]', '', x))
    
    
    ### lemmatization
    wnl = WordNetLemmatizer()
    def lemat_words(text):
        list2 = nltk.word_tokenize(text)
        lemmatized_string = ' '.join([wnl.lemmatize(words) for words in list2])  
        return lemmatized_string
    
    df_test['narrative_lem'] = df_test['narrative_ws'].apply(lemat_words)
    #print(df_test['lem_nar'])
    
    
    ### stemming
    # Initialize Python porter stemmer
    ps = PorterStemmer()
    
    def stem_words(text):
        list2 = nltk.word_tokenize(text)
        stem_string = ' '.join([ps.stem(words) for words in list2]) 
        return stem_string
    
    
    df_test['narrative_lem_stem'] = df_test['narrative_lem'].apply(stem_words)
    df_test['narrative_stem'] = df_test['narrative_ws'].apply(stem_words)
    
    #print(df_test['stem_nar'])
    return df_test






#################################################
###########  Tokens #################
#################################################

def token_sub(docs, use_bigramm, use_trigramm, use_filter_extremes, no_above, no_below, tif):
    #docs = array(df_test['test'])

    #from nltk.stem.wordnet import WordNetLemmatizer
    from nltk.tokenize import RegexpTokenizer
    
    def docs_preprocessor(docs_in):
        tokenizer = RegexpTokenizer(r'\w+')
        for idx in range(len(docs_in)):
            #docs[idx] = docs[idx].lower()  # Convert to lowercase.
            docs_in[idx] = tokenizer.tokenize(docs_in[idx])  # Split into words.
    
        # Remove numbers, but not words that contain numbers.
        docs_in = [[token for token in doc if not token.isdigit()] for doc in docs_in]
        
        # Remove words that are only one character.
        docs_in = [[token for token in doc if len(token) > 3] for doc in docs_in]
        
        # Lemmatize all words in documents.
        #lemmatizer = WordNetLemmatizer()
        #docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]
      
        return docs_in
    # Perform function on our document
    docs_out = docs_preprocessor(docs)
    #Create Biagram & Trigram Models 
    from gensim.models import Phrases
    
    # Add bigrams and trigrams to docs,minimum count 10 means only that appear 10 times or more.
    
    if use_bigramm == 1:
        bigram = Phrases(docs, min_count=10)
        for idx in range(len(docs_out)):
            for token in bigram[docs_out[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    docs[idx].append(token)
            if use_trigramm == 1:
                trigram = Phrases(bigram[docs_out])
                for token in trigram[docs_out[idx]]:
                    if '_' in token:
                        # Token is a bigram, add to document.
                        docs_out[idx].append(token)
                        
                    
    #Remove rare & common tokens 
    # Create a dictionary representation of the documents.
    mydictionary = Dictionary(docs_out)
    
    #print("dictionary = ", dictionary)
    if use_filter_extremes == 1:
        mydictionary.filter_extremes(no_below, no_above)
    
    #Create dictionary and corpus required for Topic Modeling
    corpus = [mydictionary.doc2bow(doc) for doc in docs_out]
    
    if tif == 1: 
        tfidf = models.TfidfModel(corpus)
        corpus = tfidf[corpus]    
    
    
    return mydictionary, corpus
    
  
    
    
