#!/usr/bin/python
from load_data_module import *
from sklearn.feature_extraction.text import CountVectorizer
from CMUTweetTagger import TweetTagger
import textblob as tb
from pattern.text.en import (parse, parsetree)
from pattern.text import tree


c = CountVectorizer()
tagger = TweetTagger()

t, t2 = load_data('./train.csv', "./test.csv")


tweets = list(t['tweet'])[0:10]
tags = tagger.runtagger_parse(tweets)
print tags
str_tags = []
for tweet in tags:

    str_tag = ""
    for tag in tweet:
        print tag
        if tag[2] > 0.6:
            str_tag = str_tag + tag[0]+"P"+tag[1]+" "
    str_tags.append(str_tag)
print str_tags

vct = c.fit(str_tags)
print vct.get_feature_names()

#tag.check_script_is_present()
#test = t['tweet'][1]
#print list(t['tweet'])[0]
#test = "I don't like rainy days"
#b = tb.TextBlob(test, pos_tagger=tag)
#print test
#print b.pos_tags
#print b.parse()
#result = tag.tag([test])
#print type(result),result

#s = parsetree(test,   relations=True, lemmata=False)

#for sentence in s:
#    for chunk in sentence.chunks:
#         print chunk.type, [(w.string, w.type) for w in chunk.words]