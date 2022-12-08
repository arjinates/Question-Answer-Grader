import string
import spacy
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from collections import deque

#nltk.download('stopwords')
#nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')

stop_words = set(stopwords.words('english'))

def clean_string(text):
    text = ''.join([word for word in text if word not in string.punctuation])
    text = text.lower()  
    text = ' '.join([word for word in text.split(' ') if word not in stop_words])
    return text
    #yukardaki metodun stopwords kisminda problem var cözülecek
myfile = open('QAs.txt')

correctedList = list()
for line in myfile:
    if(line[0].isdigit()):
        correctedList.append(str(TextBlob(clean_string(line)).correct()))
    else:
        correctedList[-1] += " "+str(TextBlob(clean_string(line)).correct())
myfile.close()

userProvidedAnswerList=list()
userProvidedAnswerList.append(input('Please enter true answer:'))

def cosine_sim_vectors(vec1, vec2):
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)

    return cosine_similarity(vec1, vec2)[0][0]

vectorizer = CountVectorizer()
correctedListVec = vectorizer.fit_transform(correctedList)
userVec = vectorizer.fit_transform(userProvidedAnswerList)

gradesFile = open('grades.txt','w+')

for ans in correctedList:
    ansVec =vectorizer.transform([ans])
    gradesFile.write(str((cosine_sim_vectors(userVec, ansVec)*100).round())+'\n')
gradesFile.read()
gradesFile.close()






    

