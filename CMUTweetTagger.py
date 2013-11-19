#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple Python wrapper for runTagger.sh script for CMU's Tweet Tokeniser and Part of Speech tagger: http://www.ark.cs.cmu.edu/TweetNLP/

Usage:
results=runtagger_parse(['example tweet 1', 'example tweet 2'])
results will contain a list of lists (one per tweet) of triples, each triple represents (term, type, confidence)
"""
import subprocess
import shlex
from textblob.base import BaseTagger
import textblob as tb

# The only relavent source I've found is here:
# http://m1ked.com/post/12304626776/pos-tagger-for-twitter-successfully-implemented-in
# which is a very simple implementation, my implementation is a bit more
# useful (but not much).

# NOTE this command is directly lifted from runTagger.sh


class TweetTagger(BaseTagger):

    def __init__(self, run_tagger_command="java -XX:ParallelGCThreads=2 -Xmx500m -jar ark-tweet-nlp-0.3.2/ark-tweet-nlp-0.3.2.jar"):
        self.run_tagger_command = run_tagger_command
        

    def _split_results(self, rows):
        """Parse the tab-delimited returned lines, modified from: https://github.com/brendano/ark-tweet-nlp/blob/master/scripts/show.py"""
        for line in rows:
            line = line.strip()  # remove '\n'
            if len(line) > 0:
                if line.count('\t') == 2:
                    parts = line.split('\t')
                    tokens = parts[0]
                    tags = parts[1]
                    confidence = float(parts[2].replace(",", "."))
                    yield tokens, tags, confidence


    def _call_runtagger(self,tweets):
        """Call runTagger.sh using a named input file"""

        # remove carriage returns as they are tweet separators for the stdin
        # interface
        tweets_cleaned = [tw.replace('\n', ' ') for tw in tweets]
        message = "\n".join(tweets_cleaned)

        # force UTF-8 encoding (from internal unicode type) to avoid .communicate encoding error as per:
        # http://stackoverflow.com/questions/3040101/python-encoding-for-pipe-communicate
        message = message.encode('utf-8')

        # build a list of args
        args = shlex.split(self.run_tagger_command)
        args.append('--output-format')
        args.append('conll')
        po = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # old call - made a direct call to runTagger.sh (not Windows friendly)
        #po = subprocess.Popen([run_tagger_cmd, '--output-format', 'conll'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result = po.communicate(message)
        # expect a tuple of 2 items like:
        # ('hello\t!\t0.9858\nthere\tR\t0.4168\n\n',
        # 'Listening on stdin for input.  (-h for help)\nDetected text input format\nTokenized and tagged 1 tweets (2 tokens) in 7.5 seconds: 0.1 tweets/sec, 0.3 tokens/sec\n')

        pos_result = result[0].strip('\n\n')  # get first line, remove final double carriage return
        pos_result = pos_result.split('\n\n')  # split messages by double carriage returns
        pos_results = [pr.split('\n') for pr in pos_result]  # split parts of message by each carriage return
        return pos_results


    def runtagger_parse(self, tweets):
        """Call runTagger.sh on a list of tweets, parse the result, return lists of tuples of (term, type, confidence)"""
        pos_raw_results = self._call_runtagger(tweets)
        pos_result = []
        for pos_raw_result in pos_raw_results:
            pos_result.append([x for x in self._split_results(pos_raw_result)])
        return pos_result

    def tag(self, tweet):
        print tweet
        results = self.runtagger_parse([tweet])[0]
        results = [(val[0], val[1]) for val in results]
        print results
        return results

    def str_tag(self,tweets):
        tags = self.runtagger_parse(tweets)
        str_tags = []
        for tweet in tags:
            str_tag = ""
            for tag in tweet:
                if tag[2] > 0.6:
                    str_tag = str_tag + tag[0]+"P"+tag[1]+" "
            str_tags.append(str_tag)

        return str_tags

    def check_script_is_present(self):
        """Simple test to make sure we can see the script"""
        success = False
        try:
            args = shlex.split(self.run_tagger_command)
            args.append("--help")
            po = subprocess.Popen(args, stdout=subprocess.PIPE)
            # old call - made a direct call to runTagger.sh (not Windows friendly)
            #po = subprocess.Popen([run_tagger_cmd, '--help'], stdout=subprocess.PIPE)
            while not po.poll():
                lines = [l for l in po.stdout]
            # we expected the first line of --help to look like the following:
            assert "RunTagger [options]" in lines[0]
            success = True
        except OSError as err:
            print "Caught an OSError, have you specified the correct path to runTagger.sh? We are using \"%s\". Exception: %r" % (run_tagger_cmd, repr(err))
        return success



