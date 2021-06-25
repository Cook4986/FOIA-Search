#!/usr/bin/env python
# coding: utf-8

# In[2]:


####FOIA Search - Matt Cook, 2019
###Takes pdf input and outputs abbreviated pdf with keyword highlights
###Returns Term-document matrix, Bi-gram and LDA topic analysis

import fitz
import sys
import numpy as np
import os
import textmining
import gensim
from gensim import corpora

#input/output/counts
data = "xxxx" #pdf filepath
doc = fitz.open(data)
outputFile = fitz.open()
textOut = open("xxx/TextFile" + ".txt", "wb") #text output file
numPages = doc.pageCount
pageCount = 0

#userInput 
inputString = input("Search document for keywords: ")
inputString = str(inputString)

#txt out
for page in doc:                            
    rawText = page.getText().encode("utf8") 
    textOut.write(rawText)                         
    textOut.write(b"\n-----\n") 
textOut.close()

#tokenize and bigram output
example_dir = os.path.dirname("xxx")
text_file = os.path.join(example_dir, "TextFile.txt")
textNPS = open(text_file).read()
wordsRaw = textmining.simple_tokenize(textNPS) 
wordsRaw = textmining.simple_tokenize_remove_stopwords(textNPS)
print("Hi frequency bigrams:")
print("\n")
bigrams = textmining.bigram_collocations(wordsRaw)
for bigram in bigrams[:35]:
    print(' '.join(bigram))
print("\n")

#TDM Output
tdm = textmining.TermDocumentMatrix()
tdm.add_doc(textNPS)
for row in tdm.rows(cutoff=1):
    print (row)
    
#Topic Modeling (Latent Dirichlet Allocation)
textData = []
textData.append(wordsRaw)
dictionary = corpora.Dictionary(textData) 
corpus = [dictionary.doc2bow(text) for text in textData]
NUM_TOPICS = 10
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=10)
print("lda topic modeling output:")
for topic in topics:
    print(topic)
print("\n")
print("keyword search results:")

#documentSearch  
while pageCount < numPages:
    page = doc.loadPage(pageCount)
    text = page.getText("type")
    if inputString in text:
        print ("Keyword found on page", pageCount)
        outputFile.insertPDF(doc, from_page=pageCount, to_page=pageCount, rotate=0)
        pageCount = pageCount + 1 
    elif inputString not in text: 
        print("No such keyword found on page ", pageCount)
        pageCount = pageCount + 1
        
#highlightingDeclarations
outputNP = outputFile.pageCount
print(outputNP, "pages in output file")
outputPC = 0
gold   = (1, 1, 0)
colors = {"fill": gold}

#highlightFunction
while outputPC < outputNP:
    outputPage = outputFile.loadPage(outputPC)
    r1 = outputPage.searchFor(inputString, quads = True)
    r1 = r1[0]
    annot = outputPage.addHighlightAnnot(r1)
    print("added 'Highlighting'to page", outputPC)
    outputPC = outputPC + 1
    
#close
print("\n")
print ("Outputting to PDF")
print("\n")
print ("Outputting to .txt")
outputFile.save("xxxx", expand=255) ###highlighted pdf output
outputFile.close()
doc.close()


# In[ ]:




