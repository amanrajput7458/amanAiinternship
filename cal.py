import math 
from sklearn.feature_extraction.text import TfidfVectorizer 
doc=['The cat sat on the mat','The dog played in the park','Cats and dogs are great pets']
tfidf=TfidfVectorizer()
result=tfidf.fit_transform(doc)

print(result.get_feature_names_out())
print(result)