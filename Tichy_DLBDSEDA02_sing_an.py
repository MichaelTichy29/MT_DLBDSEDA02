import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords

from numpy import array
nltk.download('stopwords')
from matplotlib import pyplot as plt
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
import re

from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models import LsiModel
from gensim import models
import pyLDAvis.gensim
import pyLDAvis
import warnings
warnings.filterwarnings("ignore")



#################################################
###########  load Dataset     #############
#################################################

# import the data as csv
df_comp = pd.read_csv('complaints_processed.csv', encoding = 'cp850')



#################################################
###########  First Impression     #############
#################################################

#####First Impression of the data
first_imp = 1
if first_imp == 1:
    # Get column names
    column_names = df_comp.columns
    print(column_names)
    
    #shape of the original dataset
    print("original comp data set")
    print(df_comp.shape)
    print("") 
    
    # Test for empty cells - > No empty cells.
    y = df_comp.isnull().sum()
    z = df_comp.shape
    #print(df_comp.shape)
    for k in range (0,z[1]): 
        print(y[k])
        print(df_comp.columns[k])
        print("")
    
    df_comp = df_comp.dropna(subset=['narrative'])
    print("without empty rows comp data set")
    print(df_comp.shape)
    


#################################################
###########  Dataset: Test or prod   #############
##########   drop na        #####################
#################################################

# drop null values
df_comp = df_comp.dropna(subset=['narrative'])
# Generate Test Data - for development
df_test = df_comp.loc[df_comp['Unnamed: 0'] < 100]
#print(df_test)
# For production: 
#df_test = df_comp



#################################################
###########  Claening     #############
#################################################

###lowercase
### stopwords
# nltk
stop = stopwords.words('english')

df_test['narrative_str'] = df_test['narrative'].apply(str)


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


##################################################
#################################################
######             Main control          ##########    
######  Choice of dataset, vect and method  ########
#################################################
#################################################

dataset = 2
if dataset == 1:    
    # original data
    df_test['test']= df_test['narrative']
elif dataset == 2:
    # without stopwords
    df_test['test'] = df_test['narrative_ws']
elif dataset == 3:
    # without stopwords and lemmatization
    df_test['test'] = df_test['narrative_lem']
elif dataset == 4:
    # without stopwords and stemming
    df_test['test'] = df_test['narrative_stem']
elif dataset == 5:
    #without stopwords, lemmatization and stemming
    df_test['test'] = df_test['narrative_lem_stem']
    
# Using Bag of words    
bow = 1 # not used - > 1 reommanded

# start the block for tokenizsation - Must be 1
use_gensim = 1   
if use_gensim == 1: 
            
    # using bigramm or trigramm (0,0) = (no, no), (1,0) = (yes,no), (1,1) = (yes, yes)
    use_bigramm = 1
    use_trigramm = 0
        
    # Using  Tfidf 
    # 0: No, 1: yes
    tif = 0
    
    # use the filter for extreme values in the corpus
    # 0: No, 1: yes
    use_filter_extremes = 0
    # real values no_above > no_below
    if use_filter_extremes == 1: 
        no_below = 0.2
        no_above = 10



# Using LSA to figure out topics
# 0: No, 1: yes
meth_lsa = 0

# Using LDA to figure out topics
# 0: No, 1: yes
meth_lda = 1



# show a vizualisation as a word cloud    
# 0: No, 1: yes
vis = 0

# show vizualisation of results
# 0: No, 1: yes
vis_res = 0

#prints the importance of the keywords for the topics.
# 0: No, 1: yes
print_top = 1

# number of topics (integer)
num_topics = 3

# coherence score umass
# 0: No, 1: yes
use_umass = 0
umass_detail = 0


#################################################
#############  Vizualisation  ################
#################################################

if vis == 1:
    # Import the wordcloud library
    from wordcloud import WordCloud
    # Join the different processed titles together.
    long_string = ','.join(list(df_test['test'].values))
    
    mywordcloud = WordCloud(background_color="white").generate(long_string)
    
    plt.imshow(mywordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    


#################################################
#################      Tokens #################
#################################################



if use_gensim == 1:
    docs = array(df_test['test'])

    #from nltk.stem.wordnet import WordNetLemmatizer
    from nltk.tokenize import RegexpTokenizer
    
    def docs_preprocessor(docs):
        tokenizer = RegexpTokenizer(r'\w+')
        for idx in range(len(docs)):
            #docs[idx] = docs[idx].lower()  # Convert to lowercase.
            docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.
    
        # Remove numbers, but not words that contain numbers.
        docs = [[token for token in doc if not token.isdigit()] for doc in docs]
        
        # Remove words that are only one character.
        docs = [[token for token in doc if len(token) > 3] for doc in docs]
        
      
        return docs
    # Perform function on our document
    docs = docs_preprocessor(docs)
    #Create Biagram & Trigram Models 
    from gensim.models import Phrases
    
    # Add bigrams and trigrams to docs,minimum count 10 means only that appear 10 times or more.
    
    if use_bigramm == 1:
        bigram = Phrases(docs, min_count=10)
        for idx in range(len(docs)):
            for token in bigram[docs[idx]]:
                if '_' in token:
                    # Token is a bigram, add to document.
                    docs[idx].append(token)
            if use_trigramm == 1:
                trigram = Phrases(bigram[docs])
                for token in trigram[docs[idx]]:
                    if '_' in token:
                        # Token is a bigram, add to document.
                        docs[idx].append(token)
                        
                    
    #Remove rare & common tokens 
    # Create a dictionary representation of the documents.
    mydictionary = Dictionary(docs)
    
    #print("dictionary = ", dictionary)
    if use_filter_extremes == 1:
        mydictionary.filter_extremes(no_below=10, no_above=0.2)
    
    #Create dictionary and corpus required for Topic Modeling
    corpus = [mydictionary.doc2bow(doc) for doc in docs]
    
    if tif == 1: 
        tfidf = models.TfidfModel(corpus)
        corpus = tfidf[corpus]    
    
    
    #print(dir(mydictionary))
    #print('Number of unique tokens: %d' % len(mydictionary))
    #print('Number of documents: %d' % len(corpus))
    #print(corpus[:2])



    
    
    


#####################################
#############   LDA   ###############
#####################################


if meth_lda == 1:
    #2 
    # Set parameters.
    #num_topics = 2
    chunksize = 500 
    passes = 20 
    iterations = 400
    eval_every = 1  
    
    # Make a index to word dictionary.
    temp = mydictionary[0]  # only to "load" the dictionary.
    id2word = mydictionary.id2token
    
        
    lda_model = LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize, \
                           alpha='auto', eta='auto', \
                           iterations=iterations, num_topics=num_topics, \
                           passes=passes, eval_every=eval_every)
    
    if print_top == 1:
        for k_top in range(0, num_topics):
           print(lda_model.print_topics(k_top))
           print('')


#####################################
#############   LSA   ###############
#####################################
if meth_lsa == 1:

    lsi_model = LsiModel(corpus, id2word=mydictionary, num_topics=num_topics)    
       
    if print_top == 1:
        for k_top in range(0, num_topics):
            print(lsi_model.print_topics(k_top))
            print('')
        

#####################################
########   coherence score   ##########
#####################################


   


# coherence score: umass - single
if use_umass == 1:
    if meth_lda == 1: 
        coherence_model_lda = CoherenceModel(model=lda_model, texts=docs, dictionary=mydictionary, coherence="u_mass")
    elif meth_lsa == 1: 
        coherence_model_lda = CoherenceModel(model=lsi_model, texts=docs, dictionary=mydictionary, coherence="u_mass")
    
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

if umass_detail == 1:
    def compute_coherence_values(dictionary, corpus, texts,coherence, limit, start=2, step=3):
        
        coherence_values = []
        model_list = []
        topic_coherence_values=[]
        for num_topics in range(start, limit+1, step):
            model=LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence=coherence)
            coherence_values.append(coherencemodel.get_coherence())
            topic_coherence_values.append(coherencemodel.get_coherence_per_topic())
    
        return model_list, coherence_values, topic_coherence_values
    
    
    limit=3; start=1; step=1;
    model_list_umass, coherence_values_umass, topic_coherence_values_umass =compute_coherence_values(dictionary=mydictionary, corpus=corpus, texts=docs,coherence="u_mass", start=start, limit=limit, step=step)
    print("Coherence score for each Topic Model (u_mass):")
    print(coherence_values_umass)
    print("--------------------")
    model=1
    for topic_coherence_value_umass in topic_coherence_values_umass:
        print(f"Coherence score for topics of Model {model}:")
        print(topic_coherence_value_umass)
        model=model+1

#################################################
########   visualisation of results   ###########
#################################################

if vis_res == 1: 
    
    vis = pyLDAvis.gensim.prepare(topic_model=lda_model, corpus=corpus, dictionary=mydictionary)
    pyLDAvis.enable_notebook()
    pyLDAvis.display(vis)
    pyLDAvis.save_html(vis, 'D:/Michael/IU/5. Semester/Election A - Advanced Data Analyst/Project/vis_of_topic.html')
 