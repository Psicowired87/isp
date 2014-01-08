import string
import re
import time
from textblob import TextBlob
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer as wnl
#from nltk import  PorterStemmer
from math import log
#import inflect
import pandas as p

import multiprocessing
from multiprocessing import Pool

class Preprocess(object):


	def __init__(self, rm_punctuation=True, rm_special=False, special_words=['mention', 'rt', 'link'], split_compound=True, translate=False, correct=False, singularize=False, lemmatize=False, abbs_file = 'tweeter_abbreviations.csv', expand_abbs=False, substitute_url=False, rm_digits=False, rm_repeated_chars=False, rm_single_chars=False):
		self.__special_words = special_words
		
		self.rm_punctuation = rm_punctuation
		self.rm_special = rm_special
		self.split_compound = split_compound
		self.translate = translate
		self.correct = correct
		self.singularize = singularize
		self.lemmatize = lemmatize
		self.expand_abbs = expand_abbs
		self.substitute_url = substitute_url
		self.rm_repeated_chars = rm_repeated_chars
		self.rm_single_chars = rm_single_chars
		self.rm_digits = rm_digits

		# Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).
		self.words = open("words-by-frequency.txt").read().split()
		self.wordcost = dict((k, log((i+1)*log(len(self.words)))) for i,k in enumerate(self.words))
		self.maxword = max(len(x) for x in self.words)

		# punctuation
		# excluding \'
		#self.not_letters_or_digits = u'!"#%()*+,-./:;<=>?@[\]^_`{|}~'
		self.not_letters_or_digits = u'!"#%()\'*+,-./:;<=>?@[\]^_`{|}~'


		self.tweet_abbs = p.read_csv('tweeter_abbreviations.csv')

		#remove words of length 1
		self.remove_letter = re.compile(r'\W*\b\w{1,1}\b')
		#remove_letter.sub('', anytext)

		self.url = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')

		tab = p.read_csv(abbs_file)
		self.abbs = tab['abb'].tolist()
		self.explanations = tab['plain'].tolist()


		tab = p.read_csv('CorrectionsList/Corrections_N.csv')
		self.dictN = dict(zip(tab['abb'].tolist(), tab['plain'].tolist()))
		
		tab = p.read_csv('CorrectionsList/Corrections_O.csv')
		self.dictO = dict(zip(tab['abb'].tolist(), tab['plain'].tolist()))
		
		tab = p.read_csv('CorrectionsList/Corrections_^.csv')
		self.dictHat = dict(zip(tab['abb'].tolist(), tab['plain'].tolist()))
		
		tab = p.read_csv('CorrectionsList/Corrections_S.csv')
		self.dictS = dict(zip(tab['abb'].tolist(), tab['plain'].tolist()))
		
		tab = p.read_csv('CorrectionsList/Corrections_Z.csv')
		self.dictZ = dict(zip(tab['abb'].tolist(), tab['plain'].tolist()))
		
		tab = p.read_csv('CorrectionsList/Corrections_V.csv')
		self.dictV = dict(zip(tab['abb'].tolist(), tab['plain'].tolist()))
		
		tab = p.read_csv('CorrectionsList/Corrections_A.csv')
		self.dictA = dict(zip(tab['abb'].tolist(), tab['plain'].tolist()))
		
		tab = p.read_csv('CorrectionsList/Corrections_R.csv')
		self.dictR = dict(zip(tab['abb'].tolist(), tab['plain'].tolist()))
		
		tab = p.read_csv('CorrectionsList/Corrections_!.csv')
		self.dictInterjection = dict(zip(tab['abb'].tolist(), tab['plain'].tolist()))

		tab = p.read_csv('CorrectionsList/Corrections_D.csv')
		self.dictD = dict(zip(tab['abb'].tolist(), tab['plain'].tolist()))

		tab = p.read_csv('CorrectionsList/Corrections_P.csv')
		self.dictP = dict(zip(tab['abb'].tolist(), tab['plain'].tolist()))
		
		tab = p.read_csv('CorrectionsList/Corrections_&.csv')
		self.dictAnd = dict(zip(tab['abb'].tolist(), tab['plain'].tolist()))
		
		tab = p.read_csv('CorrectionsList/Corrections_T.csv')
		self.dictT = dict(zip(tab['abb'].tolist(), tab['plain'].tolist()))

		tab = p.read_csv('CorrectionsList/Corrections_X.csv')
		self.dictX = dict(zip(tab['abb'].tolist(), tab['plain'].tolist()))

		tab = p.read_csv('CorrectionsList/Corrections_hash.csv')
		self.dictHash = dict(zip(tab['abb'].tolist(), tab['plain'].tolist()))
		
		tab = p.read_csv('CorrectionsList/Corrections_At.csv')
		self.dictAt = dict(zip(tab['abb'].tolist(), tab['plain'].tolist()))
		
		tab = p.read_csv('CorrectionsList/Corrections_G.csv')
		self.dictG = dict(zip(tab['abb'].tolist(), tab['plain'].tolist()))

		tab = p.read_csv('CorrectionsList/Corrections_L.csv')
		self.dictL = dict(zip(tab['abb'].tolist(), tab['plain'].tolist()))

	def __call__(self, aTweet):

		newTweet = [aTweet]
		newTweet = self.run(newTweet)

		return newTweet[0].split()



	def preprocess_tagged(self, tagged_tweet):
		newTweet = ''
		for (w, tag) in tagged_tweet:
			if tag=='N':
				newTweet = newTweet + self.dictN.get(w,w) + ' '
			elif tag=='O':
				newTweet = newTweet + self.dictO.get(w,w) + ' '
			elif tag=='^':
				newTweet = newTweet + self.dictHat.get(w,w) + ' '
			elif tag=='S':
				newTweet = newTweet + self.dictS.get(w,w) + ' '
			elif tag=='Z':
				newTweet = newTweet + self.dictZ.get(w,w) + ' '
			elif tag=='V':
				newTweet = newTweet + self.dictV.get(w,w) + ' '
			elif tag=='A':
				newTweet = newTweet + self.dictA.get(w,w) + ' '
			elif tag=='R':
				newTweet = newTweet + self.dictR.get(w,w) + ' '
			elif tag=='!':
				newTweet = newTweet + self.dictInterjection.get(w,w) + ' '
			elif tag=='D':
				newTweet = newTweet + self.dictD.get(w,w) + ' '
			elif tag=='P':
				newTweet = newTweet + self.dictP.get(w,w) + ' '
			elif tag=='&':
				newTweet = newTweet + self.dictAnd.get(w,w) + ' '
			elif tag=='T':
				newTweet = newTweet + self.dictT.get(w,w) + ' '
			elif tag=='X':
				newTweet = newTweet + self.dictX.get(w,w) + ' '
			elif tag=='#':
				newTweet = newTweet + self.dictHash.get(w,w) + ' '
			elif tag=='@':
				newTweet = newTweet + self.dictAt.get(w,w) + ' '
			elif tag=='~':
				newTweet = newTweet + ''
			elif tag=='U':
				newTweet = newTweet + ''
			elif tag=='E':
				newTweet = newTweet + ''
			elif tag=='$':
				newTweet = newTweet + ''
			elif tag==',':
				newTweet = newTweet + ''
			elif tag=='G':
				newTweet = newTweet + self.dictG.get(w,w) + ' '
			elif tag=='L':
				newTweet = newTweet + self.dictL.get(w,w) + ' '

		return newTweet


	def runTaggedParallel(self, tagged_tweets):

		l = len(tagged_tweets)*[None]
		for i, t in enumerate(tagged_tweets):
			l[i] =  self.preprocess_tagged(tagged_tweets[i])

		return l


	def runParallel(self, tweets):

		l_tweets = [[t] for t in tweets]

		aPool = Pool()
		l =  aPool.map(self.run, l_tweets)

		l = [item for sublist in l for item in sublist]

		aPool.close()
		aPool.join()

		return l



	def run(self, tweets):

		l = len(tweets)*[None]

		for i, t in enumerate(tweets):
			t1=t.lower()

			if self.substitute_url:
				t1 = self.substitute_urls([t1])[0]
			if self.rm_repeated_chars:
				t1 = self.remove_repeated_characters(t1)
			if self.expand_abbs:
				t1 = self.extend_abbreviations([t1])[0]
			if self.rm_punctuation:
				t1 = self.remove_punctuation(t1)
			if self.rm_digits:
				t1 = self.remove_digits(t1)
			if self.rm_single_chars:
				t1 = self.remove_single_characters(t1)
			if self.rm_special:
				t1 = self.remove_special_words_tweet(t1)
			if self.split_compound:
				t1 = self.split_compound_words(t1)
			if self.correct:
				t1 = self.correct_spelling_tweet(t1)
			if self.translate:
				t1 = self.translate_tweet_to_EN(t1)
			if self.singularize:
				t1 = self.singularize_tweet(t1)
			if self.lemmatize:
				t1 = self.lemmatize_nouns_in_tweet(t1)

			#remove more than one space
			l[i] = re.sub(' +', ' ', t1)

		return l


	def remove_single_characters(self, tweet):
		#print('Removing punctuation..., end= ' ')
		#timestamp1 = time.time()

		l = self.remove_letter.sub('', tweet)

		#timestamp2 = time.time()
		#print('{0:.2f} seconds'.format(timestamp2 - timestamp1))

		return l


	def remove_repeated_characters(self, tweet):
		#print('Removing punctuation..., end= ' ')
		#timestamp1 = time.time()

		#remove repeated characters leaving 2
		l = re.sub(r'(.)\1+', r'\1\1', tweet)

		#timestamp2 = time.time()
		#print('{0:.2f} seconds'.format(timestamp2 - timestamp1))

		return l


	def remove_digits(self, tweet):
		#print('Removing punctuation..., end= ' ')
		#timestamp1 = time.time()

		# remove numbers
		l = ''.join(i for i in tweet if not i.isdigit())

		#timestamp2 = time.time()
		#print('{0:.2f} seconds'.format(timestamp2 - timestamp1))

		return l


	def remove_punctuation(self, tweet):
		#print('Removing punctuation..., end= ' ')
		#timestamp1 = time.time()

		l = self.translate_non_alphanumerics(tweet)

		#timestamp2 = time.time()
		#print('{0:.2f} seconds'.format(timestamp2 - timestamp1))

		return l



	def translate_non_alphanumerics(self, to_translate, translate_to=u' '):
		translate_table = dict((ord(char), translate_to) for char in self.not_letters_or_digits)
		return to_translate.translate(translate_table)



	def remove_special_words_tweet(self, tweet):
		#print('Removing special words..., end= ' ')
		#timestamp1 = time.time()

		l = tweet
		for sw in self.__special_words:
			l = re.sub(re.escape(sw), '', l)
		
		#timestamp2 = time.time()
		#print('{0:.2f} seconds'.format(timestamp2 - timestamp1))

		return l
		


	def singularize_tweet(self, tweet):
		#print('Singularize tweets..., end= ' ')
		#timestamp1 = time.time()

		t = TextBlob(tweet)
		l = ''
		for w in t.words:
			l = l + w.singularize() + ' '

		#timestamp2 = time.time()
		#print('{0:.2f} seconds'.format(timestamp2 - timestamp1))
			
		return l


	def translate_tweet_to_EN(self, tweet):
		#print('Translation..., end= ' ')
		#timestamp1 = time.time()

		l = tweet
		t = TextBlob(tweet)
		if t.detect_language() != u'en':
			l = t.translate().raw_sentences[0]
		
		#print( n, 'translated')
		#timestamp2 = time.time()
		#print('{0:.2f} seconds'.format(timestamp2 - timestamp1))

		return l


	def split_compound_words(self, tweet):
		#print('Split compound words..., end= ' ')
		#timestamp1 = time.time()

		l = ''
		for w in tweet.split():
			if len(w)>5 and not wn.synsets(w):
				l += ' '.join(self.infer_spaces(w)) + ' '
			else:	
				l += w + ' '

		#timestamp2 = time.time()
		#print('{0:.2f} seconds'.format(timestamp2 - timestamp1))
		
		return l


	def correct_spelling_tweet(self, tweet):
		#print('Spell correction..., end= ' ')
		#timestamp1 = time.time()
	
		l = TextBlob(tweet).correct().raw_sentences[0]
	
		#timestamp2 = time.time()
		#print('{0:.2f} seconds'.format(timestamp2 - timestamp1))
			
		return l


	def infer_spaces(self, s):
		"""Uses dynamic programming to infer the location of spaces in a string
		without spaces."""

		# Find the best match for the i first characters, assuming cost has
		# been built for the i-1 first characters.
		# Returns a pair (match_cost, match_length).
		def best_match(i):
			candidates = enumerate(reversed(cost[max(0, i-self.maxword):i]))
			return min((c + self.wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)

		# Build the cost array.
		cost = [0]
		for i in range(1,len(s)+1):
			c,k = best_match(i)
			cost.append(c)

		# Backtrack to recover the minimal-cost string.
		out = []
		i = len(s)
		while i>0:
			c,k = best_match(i)
			assert c == cost[i]
			out.append(s[i-k:i])
			i -= k

		return reversed(out)



	def lemmatize_nouns_in_tweet(self, tweet):
		#print('Nouns lemmatization..., end= ' ')
		#timestamp1 = time.time()
		

		l = ' '.join([wnl().lemmatize(t) for t in tweet.split()])
	
		#timestamp2 = time.time()
		#print('{0:.2f} seconds'.format(timestamp2 - timestamp1))
			
		return l


	def extend_abbreviations(self, tweets):
		#total=0
		newtweets = [t.lower() for t in tweets]

		for i, abb in enumerate(self.abbs):
			#n=0
		
			#timestamp1 = time.time()

			for j, t in enumerate(tweets):
				if bool(re.search('(^|[^a-z0-9\'])' + self.abbs[i] + '($|[^a-z0-9\'])', t)):
					newtweets[j] = re.sub(r"(^|[^a-z0-9\'])%s($|[^a-z0-9\'])" % abb, ' '+self.explanations[i]+' ', t)
					#n+=1

			#total+=n

			#timestamp2 = time.time()
			#print('{0:10} : {1:-5}\t\t {2:.2f} seconds'.format(abb, n, timestamp2 - timestamp1))


		#print('Total: {0}'.format(total))	

		return newtweets


	def substitute_urls(self, tweets, new_str='{link}'):
		l = len(tweets)*[None]

		for i, t in enumerate(tweets):
			l[i] = self.url.sub(new_str,  t)

		return l
