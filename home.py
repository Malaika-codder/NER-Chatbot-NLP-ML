import nltk
from nltk.tag import RegexpTagger
from nltk.tokenize import word_tokenize

nltk.download("punkt_tab")

pattern=[
    (r'.*ing$','VBG'),
    (r'.*ed$','VBD'),
    (r'.*','NN')]

rule_tags=RegexpTagger(pattern)
text="She is playing video game"

token=word_tokenize(text)
tag=rule_tags.tag(token)

print("Tags : ",tag)
