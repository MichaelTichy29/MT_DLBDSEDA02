import pandas as pd
import numpy as np
from numpy import array
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.models import LsiModel

from Tichy_DLBDSEDA02_sub import *

list_dataset = []
list_bigramm = []
list_trigramm = []
list_fe = []
list_tif = []
list_numtop = []
list_coherence = []


meth_lda = 1
meth_lsa = 0

def Auswertung():
    #load data
    df_comp = pd.read_csv('complaints_processed.csv', encoding = 'cp850')
    #for test:
    df_test = df_comp.loc[df_comp['Unnamed: 0'] < 1000]
    df_test = cleaning_sub(df_test)
   
    # for production: 
    #df_test = cleaning_sub(df_comp)
    
    
    #default:
    no_low = 0.2
    no_ab = 10
    
    for dataset in range(5):
        for use_bigramm in range(2):
            for use_trigramm in range(use_bigramm+1):
                for use_fe in range(2):
                    if use_fe == 1:
                        no_low = 0.2
                        no_ab = 10    
                    for tif in range(2):
                        if dataset == 0:    
                            # original data
                            docs= array(df_test['narrative'])
                        elif dataset == 1:
                            # without stopwords
                            docs = array(df_test['narrative_ws'])
                        elif dataset == 2:
                            # without stopwords and lemmatization
                            docs = array(df_test['narrative_lem'])
                        elif dataset == 3:
                            # without stopwords and stemming
                            docs = array(df_test['narrative_stem'])
                        elif dataset == 4:
                            #without stopwords, lemmatization and stemming
                            docs = array(df_test['narrative_lem_stem'])
                        
                        mydictionary, corpus = token_sub(docs, use_bigramm, use_trigramm, use_fe, no_ab, no_low, tif)
                        for num_topics in range(2,10):
                            chunksize = 500 
                            passes = 20 
                            iterations = 400
                            eval_every = 1  
                            
                            # Make a index to word dictionary.
                            temp = mydictionary[0]  # only to "load" the dictionary.
                            id2word = mydictionary.id2token
                            
                            if meth_lda == 1: 
                                lda_model = LdaModel(corpus=corpus, id2word=id2word, chunksize=chunksize, \
                                                       alpha='auto', eta='auto', \
                                                       iterations=iterations, num_topics=num_topics, \
                                                       passes=passes, eval_every=eval_every)
    
                                coherence_model_lda = CoherenceModel(model=lda_model, texts=docs, dictionary=mydictionary, coherence="u_mass")
                                coherence_lda = coherence_model_lda.get_coherence()
                            elif meth_lsa == 1:     
                                                        
                                lsi_model = LsiModel(corpus, id2word=mydictionary, num_topics=num_topics)    
                                coherence_model_lda = CoherenceModel(model=lsi_model, texts=docs, dictionary=mydictionary, coherence="u_mass")
                                coherence_lda = coherence_model_lda.get_coherence()
                            
                                
                            print('\nCoherence Score: ', coherence_lda)
                            list_dataset.append(dataset)
                            list_bigramm.append(use_bigramm)
                            list_trigramm.append(use_trigramm)
                            list_fe.append(use_fe)
                            list_tif.append(tif)
                            list_numtop.append(num_topics)
                            list_coherence.append(coherence_lda)

    
    return                 
                            
Auswertung = Auswertung()

# list to dataframe
dfres = pd.DataFrame([list_dataset, list_bigramm, list_trigramm, list_fe, list_tif, list_numtop, list_coherence],
                     index=["Dataset", "bigramm", "trigramm", "fe", "tif", "num_top", "coherence"])
dfres = dfres.transpose()

if meth_lda == 1:
    # writing Dataframe to Excel
    datatoexcel = pd.ExcelWriter('result_lda.xlsx')
elif meth_lsa == 1: 
    # writing Dataframe to Excel
    datatoexcel = pd.ExcelWriter('result_lsa.xlsx')

# write DataFrame to excel
dfres.to_excel(datatoexcel)
# save the excel
datatoexcel.close()
