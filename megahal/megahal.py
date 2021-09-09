# Copyright (c) 2010, Chris Jones
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# - Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Python implementation of megahal markov bot"""

import math
import os
import random
import re
import shelve
from copy import deepcopy
from time import time
from typing import TYPE_CHECKING, cast

from megahal.util import capitalize, split_list_to_sentences

if TYPE_CHECKING:
    from typing import List, Optional

__author__ = 'Chris Jones <cjones@gruntle.org>, Robert Huselius'
__license__ = 'BSD'
__all__ = [
    'MegaHAL', 'Dictionary', 'Tree', 'DEFAULT_ORDER',
    'DEFAULT_BRAINFILE', 'DEFAULT_TIMEOUT'
]

DEFAULT_ORDER = 5
DEFAULT_BRAINFILE = os.path.join(
    os.environ.get('HOME', ''), '.pymegahal-brain')
DEFAULT_TIMEOUT = 25.0

API_VERSION = '1.0'
END_WORD = '<FIN>'
ERROR_WORD = '<ERROR>'

DEFAULT_BANWORDS = ['A', 'ABILITY', 'ABLE', 'ABOUT', 'ABSOLUTE', 'ABSOLUTELY', 'ACROSS', 'ACTUAL', 'ACTUALLY', 'AFTER',
                    'AGAIN', 'AGAINST', 'AGO', 'AGREE', 'ALL', 'ALMOST', 'ALONG', 'ALREADY', 'ALTHOUGH', 'ALWAYS',
                    'AN', 'AND', 'ANOTHER', 'ANY', 'ANYHOW', 'ANYTHING', 'ANYWAY', 'ARE', "AREN'T", 'AROUND', 'AS',
                    'AWAY', 'BACK', 'BAD', 'BE', 'BEEN', 'BEFORE', 'BEHIND', 'BEING', 'BELIEVE', 'BELONG', 'BEST',
                    'BETWEEN', 'BIG', 'BIGGER', 'BIGGEST', 'BIT', 'BOTH', 'BUDDY', 'BUT', 'BY', 'CALL', 'CALLED',
                    'CAME', 'CAN', "CAN'T", 'CANNOT', 'CARE', 'CARING', 'CASE', 'CATCH', 'CAUGHT', 'CERTAIN',
                    'CHANGE', 'CLOSE', 'CLOSER', 'COME', 'COMING', 'COMMON', 'CONSTANT', 'CONSTANTLY', 'COULD',
                    'DAY', 'DAYS', 'DERIVED', 'DESCRIBE', 'DESCRIBES', 'DETERMINE', 'DETERMINES', 'DID', "DIDN'T",
                    'DOES', "DOESN'T", 'DOING', "DON'T", 'DONE', 'DOUBT', 'DOWN', 'EACH', 'EARLIER', 'EARLY', 'ELSE',
                    'ESPECIALLY', 'EVEN', 'EVER', 'EVERY', 'EVERYBODY', 'EVERYONE', 'EVERYTHING', 'FACT', 'FAIR',
                    'FAR', 'FELLOW', 'FEW', 'FIND', 'FINE', 'FOR', 'FORM', 'FOUND', 'FROM', 'FULL', 'FURTHER', 'GAVE',
                    'GETTING', 'GIVE', 'GIVEN', 'GIVING', 'GO', 'GOING', 'GONE', 'GOOD', 'GOT', 'GOTTEN', 'GREAT',
                    'HAS', "HASN'T", 'HAVE', "HAVEN'T", 'HAVING', 'HELD', 'HERE', 'HIGH', 'HOLD', 'HOLDING', 'HOW',
                    'IN', 'INDEED', 'INSIDE', 'INSTEAD', 'INTO', 'IS', "ISN'T", 'IT', "IT'S", 'ITS', 'JUST', 'KEEP',
                    'KNEW', 'KNOW', 'KNOWN', 'LARGE', 'LARGER', 'LARGETS', 'LAST', 'LATE', 'LATER', 'LEAST', 'LESS',
                    "LET'S", 'LEVEL', 'LIKES', 'LITTLE', 'LONG', 'LONGER', 'LOOK', 'LOOKED', 'LOOKING', 'LOOKS', 'LOW',
                    'MAKE', 'MAKING', 'MANY', 'MATE', 'MAY', 'MAYBE', 'MEAN', 'MEET', 'MENTION', 'MERE', 'MIGHT',
                    'MORE', 'MORNING', 'MOST', 'MOVE', 'MUCH', 'MUST', 'NEAR', 'NEARER', 'NEVER', 'NEXT', 'NICE',
                    'NONE', 'NOON', 'NOONE', 'NOT', 'NOTE', 'NOTHING', 'NOW', 'OBVIOUS', 'OF', 'OFF', 'ON', 'ONCE',
                    'ONTO', 'OPINION', 'OR', 'OTHER', 'OUR', 'OUT', 'OVER', 'OWN', 'PART', 'PARTICULAR',
                    'PERHAPS', 'PERSON', 'PIECE', 'PLACE', 'PLEASANT', 'PLEASE', 'POPULAR', 'PREFER', 'PRETTY', 'PUT',
                    'REAL', 'REALLY', 'RECEIVE', 'RECEIVED', 'RECENT', 'RECENTLY', 'RELATED', 'RESULT', 'RESULTING',
                    'SAID', 'SAME', 'SAW', 'SAY', 'SAYING', 'SEE', 'SEEM', 'SEEMED', 'SEEMS', 'SEEN', 'SELDOM',
                    'SET', 'SEVERAL', 'SHALL', 'SHORT', 'SHORTER', 'SHOULD', 'SHOW', 'SHOWS', 'SIMPLE', 'SIMPLY',
                    'SO', 'SOME', 'SOMEONE', 'SOMETHING', 'SOMETIME', 'SOMETIMES', 'SOMEWHERE', 'SORT', 'SORTS',
                    'SPENT', 'STILL', 'STUFF', 'SUCH', 'SUGGEST', 'SUGGESTION', 'SUPPOSE', 'SURE', 'SURELY',
                    'SURROUNDS', 'TAKE', 'TAKEN', 'TAKING', 'TELL', 'THAN', 'THANK', 'THANKS', 'THAT', "THAT'S",
                    'THE', 'THEIR', 'THEM', 'THEN', 'THERE', 'THEREFORE', 'THESE', 'THEY', 'THING', 'THINGS', 'THIS',
                    'THOUGH', 'THOUGHTS', 'THOUROUGHLY', 'THROUGH', 'TINY', 'TO', 'TODAY', 'TOGETHER', 'TOLD',
                    'TOO', 'TOTAL', 'TOTALLY', 'TOUCH', 'TRY', 'TWICE', 'UNDER', 'UNDERSTAND', 'UNDERSTOOD', 'UNTIL',
                    'US', 'USED', 'USING', 'USUALLY', 'VARIOUS', 'VERY', 'WANT', 'WANTED', 'WANTS', 'WAS', 'WATCH',
                    'WAYS', 'WE', "WE'RE", 'WELL', 'WENT', 'WERE', 'WHAT', "WHAT'S", 'WHATEVER', 'WHATS', 'WHEN',
                    "WHERE'S", 'WHICH', 'WHILE', 'WHILST', 'WHO', "WHO'S", 'WHOM', 'WILL', 'WISH', 'WITH', 'WITHIN',
                    'WONDERFUL', 'WORSE', 'WORST', 'WOULD', 'WRONG', 'YESTERDAY', 'YET']

DEFAULT_AUXWORDS = ['DISLIKE', 'HE', 'HER', 'HERS', 'HIM', 'HIS', 'I', "I'D", "I'LL", "I'M", "I'VE", 'LIKE', 'ME',
                    'MY', 'MYSELF', 'ONE', 'SHE', 'THREE', 'TWO', 'YOU', "YOU'D", "YOU'LL", "YOU'RE", "YOU'VE", 'YOUR',
                    'YOURSELF']

DEFAULT_SWAPWORDS = {"YOU'RE": "I'M", "YOU'D": "I'D", 'HATE': 'LOVE', 'YOUR': 'MY', "I'LL": "YOU'LL", 'NO': 'YES',
                     'WHY': 'BECAUSE', 'YOU': 'ME', 'LOVE': 'HATE', 'I': 'YOU', 'MINE': 'YOURS', 'YOURSELF': 'MYSELF',
                     'DISLIKE': 'LIKE', "I'M": "YOU'RE", 'ME': 'YOU', 'MYSELF': 'YOURSELF', 'LIKE': 'DISLIKE',
                     "I'D": "YOU'D", "YOU'VE": "I'VE", 'YES': 'NO', 'MY': 'YOUR'}


class Reply:
    def __init__(self, text, surprise=0.0):
        self.text = text
        self.surprise = float(surprise)
        if self.surprise:
            self.rating = self.surprise
        else:
            self.rating = 0.0

    def __repr__(self):
        return "<Reply: %s>" % self.text

    def __str__(self):
        return self.text


class Tree(object):
    def __init__(self, symbol=0):
        self.symbol = symbol
        self.usage = 0
        self.count = 0
        self.children = []

    def add_symbol(self, symbol):
        node = cast("Tree", self.get_child(symbol))
        node.count += 1
        self.usage += 1
        return node

    def get_child(self, symbol, add=True):
        child: "Optional[Tree]"
        for child in self.children:
            if child.symbol == symbol:
                break
        else:
            if add:
                child = Tree(symbol)
                self.children.append(child)
            else:
                child = None
        return child


class Dictionary(list):
    def add_word(self, word):
        try:
            return self.index(word)
        except ValueError:
            self.append(word)
            return len(self) - 1

    def find_word(self, word):
        try:
            return self.index(word)
        except ValueError:
            return 0


class Brain(object):
    def __init__(self, order=None, file=None, timeout=None, banwords=None):
        self.timeout = timeout or DEFAULT_TIMEOUT
        self.db = shelve.open(file or DEFAULT_BRAINFILE, writeback=True)
        self.init_db(order=order, banwords=banwords)
        self.closed = False

    def init_db(self, clear=False, order=None, banwords=None):
        if clear:
            banwords = banwords or deepcopy(self.db["banwords"])
            self.db.clear()
        if self.db.setdefault('api', API_VERSION) != API_VERSION:
            raise ValueError(
                'This brain has an incompatible api version: %d != %d'
                % (self.db['api'], API_VERSION)  # type: ignore[str-format]
            )
        order = order or DEFAULT_ORDER
        if self.db.setdefault('order', order) != order:
            raise ValueError('This brain already has an order of %d' % self.db['order'])
        self.forward = self.db.setdefault('forward', Tree())
        self.backward = self.db.setdefault('backward', Tree())
        self.dictionary = self.db.setdefault('dictionary', Dictionary())
        self.error_symbol = self.dictionary.add_word(ERROR_WORD)
        self.end_symbol = self.dictionary.add_word(END_WORD)
        banwords = banwords or DEFAULT_BANWORDS
        self.banwords = self.db.setdefault('banwords', Dictionary(banwords))
        self.auxwords = self.db.setdefault('auxwords', Dictionary(DEFAULT_AUXWORDS))
        self.swapwords = self.db.setdefault('swapwords', DEFAULT_SWAPWORDS)

    @property
    def order(self) -> int:
        return self.db['order']

    @staticmethod
    def get_words_from_phrase(phrase):
        phrase = phrase.upper()
        words = []
        if phrase:
            offset = 0

            def boundary(string: str, position: int) -> bool:
                if position == 0:
                    boundary = False
                elif position == len(string):
                    boundary = True
                elif (string[position] == "'" and
                      string[position - 1].isalpha() and
                      string[position + 1].isalpha()):
                    boundary = False
                elif (position > 1 and
                      string[position - 1] == "'" and
                      string[position - 2].isalpha() and
                      string[position].isalpha()):
                    boundary = False
                elif (string[position].isalpha() and
                      not string[position - 1].isalpha()):
                    boundary = True
                elif (not string[position].isalpha() and
                      string[position - 1].isalpha()):
                    boundary = True
                elif string[position].isdigit() != string[position - 1].isdigit():
                    boundary = True
                else:
                    boundary = False
                return boundary

            while True:
                try:
                    if boundary(phrase, offset):
                        word, phrase = phrase[:offset], phrase[offset:]
                        words.append(word)
                        if not phrase:
                            break
                        offset = 0
                    else:
                        offset += 1
                except Exception as e:
                    print(phrase, offset)
                    raise e
            if words[-1][0].isalnum():
                words.append('.')
            elif words[-1][-1] not in '!.?':
                words[-1] = '.'
        return words

    def communicate(self, phrase, learn=True, reply=True, max_length=None, timeout=None):
        words = self.get_words_from_phrase(phrase)
        if learn:
            self.learn(words)
        if reply:
            return self.get_reply(words, max_length=max_length, timeout=timeout)
        return None

    def get_context(self, tree):

        class Context(dict):
            # pylint: disable=no-self-argument
            def __enter__(context):
                context.used_key = False
                context[0] = tree
                return context

            def __exit__(context, *exc_info):
                context.update(self.end_symbol)

            @property
            def root(context):
                return context[0]

            def update(context, symbol):
                for i in range(self.order + 1, 0, -1):
                    node = context.get(i - 1)
                    if node is not None:
                        context[i] = node.add_symbol(symbol)

            def seed(context, keys):
                if keys:
                    i = random.randrange(len(keys))
                    for key in keys[i:] + keys[:i]:
                        if key not in self.auxwords:
                            try:
                                return self.dictionary.index(key)
                            except ValueError:
                                pass
                if context.root.children:
                    return random.choice(context.root.children).symbol
                return 0

            def babble(context, keys, replies):
                for i in range(self.order + 1):
                    if context.get(i) is not None:
                        node = context[i]
                if not node.children:
                    return 0
                i = random.randrange(len(node.children))
                count = random.randrange(node.usage)
                symbol = 0
                while count >= 0:
                    symbol = node.children[i].symbol
                    word = self.dictionary[symbol]
                    if word in keys and (context.used_key or word not in self.auxwords):
                        context.used_key = True
                        break
                    count -= node.children[i].count
                    if i >= len(node.children) - 1:
                        i = 0
                    else:
                        i = i + 1
                return symbol

        return Context()

    def learn(self, words):
        if len(words) > self.order:
            with self.get_context(self.forward) as context:
                for word in words:
                    context.update(self.dictionary.add_word(word))
            with self.get_context(self.backward) as context:
                for word in reversed(words):
                    context.update(self.dictionary.index(word))

    def get_reply(self, words, max_length=None, timeout=None):
        replies = self.get_replies(words, max_length=max_length, timeout=timeout)
        if replies:
            return replies[0]
        return None

    def get_replies(self, words, max_length=None, timeout=None):
        replies: "List[Reply]" = []
        timeout = self.timeout if timeout is None else float(timeout)
        if words:
            # timeout[0] is for generating dummy reply, timeout[1] for
            # generating proper reply.
            timeouts = (timeout * 0.2, timeout * 0.8)
        else:
            # If there is nothing to reply to, only dummy reply will be
            # generated.
            timeouts = (timeout, 0.0)
        dummy_reply = None

        def trim_reply(strings: "List[str]"):
            # Trim to max number of whole sentences to fit in max_length
            assert isinstance(max_length, int)
            sentences = split_list_to_sentences(strings)
            for i in range(len(sentences), 0, -1):
                if max_length and len("".join(["".join(s) for s in sentences[:i]]).strip()) <= max_length:
                    return [w for s in sentences[:i] for w in s]
                return []

        keywords = self.make_keywords(words)
        basetime = time()
        while time() - basetime < timeouts[0]:
            dummy = self.generate_replywords()
            if max_length:
                dummy = trim_reply(dummy)
            reply_str = "".join(dummy)
            surprise = self.evaluate_reply(keywords, dummy) if words else 0.0
            if dummy and surprise > -1.0:
                break
        if dummy:
            dummy_reply = Reply(
                capitalize(reply_str),
                surprise=surprise
            )
        if words:
            # Only go through this trouble if we're actually responding to
            # something
            basetime = time()
            while time() - basetime < timeouts[1]:
                reply = self.generate_replywords(keywords)
                if max_length:
                    reply = trim_reply(reply)
                surprise = self.evaluate_reply(keywords, reply)
                reply_str = "".join(reply)
                if reply_str and surprise > -1.0:
                    replies.append(Reply(
                        capitalize(reply_str),
                        surprise=surprise
                    ))
        if replies:
            return sorted(replies, key=lambda r: r.rating, reverse=True)
        elif dummy_reply:
            return [dummy_reply]
        return []

    def evaluate_reply(self, keys, words):
        state = {'num': 0, 'entropy': 0.0}
        if words:
            def evaluate(node, words):
                with self.get_context(node) as context:
                    for word in words:
                        symbol = self.dictionary.index(word)
                        context.update(symbol)
                        if word in keys:
                            prob = 0.0
                            count = 0
                            state['num'] += 1
                            for j in range(self.order):
                                node = context.get(j)
                                if node is not None:
                                    child = node.get_child(symbol, add=False)
                                    if child:
                                        prob += float(child.count) / node.usage
                                    count += 1
                            if count:
                                state['entropy'] -= math.log(prob / count)

            evaluate(self.forward, words)
            evaluate(self.backward, reversed(words))

            if state['num'] >= 8:
                state['entropy'] /= math.sqrt(state['num'] - 1)
            if state['num'] >= 16:
                state['entropy'] /= state['num']
        return state['entropy']

    def generate_replywords(self, keys=None):
        if keys is None:
            keys = []
        replies: "List[str]" = []
        with self.get_context(self.forward) as context:
            start = True
            while True:
                if start:
                    symbol = context.seed(keys)
                    start = False
                else:
                    symbol = context.babble(keys, replies)
                if symbol in (self.error_symbol, self.end_symbol):
                    break
                replies.append(self.dictionary[symbol])
                context.update(symbol)
        with self.get_context(self.backward) as context:
            if replies:
                for i in range(min([(len(replies) - 1), self.order]), -1, -1):
                    context.update(self.dictionary.index(replies[i]))
            while True:
                symbol = context.babble(keys, replies)
                if symbol in (self.error_symbol, self.end_symbol):
                    break
                replies.insert(0, self.dictionary[symbol])
                context.update(symbol)

        return replies

    def make_keywords(self, words):
        keys = Dictionary()
        for word in words:
            try:
                word = self.swapwords[word]
            except KeyError:
                pass
            if (self.dictionary.find_word(word) != self.error_symbol and word[0].isalnum() and
                    word not in self.banwords and word not in self.auxwords and word not in keys and
                    len(word) > 1):
                keys.append(word)

        if keys:
            for word in words:
                try:
                    word = self.swapwords[word]
                except KeyError:
                    pass
                if (self.dictionary.find_word(word) != self.error_symbol and word[0].isalnum() and
                        word in self.auxwords and word not in keys):
                    keys.append(word)

        return keys

    def add_key(self, keys, word):
        # Never used?!
        if (len(word) > 1 and self.dictionary.find_word(word) != self.error_symbol and
            self.banwords.find_word(word) == self.error_symbol and
                self.auxwords.find_word(word) == self.error_symbol):
            keys.add_word(word)

    def sync(self):
        self.db.sync()

    def close(self):
        if not self.closed:
            print('Closing database')
            self.db.close()
            self.closed = True

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class MegaHAL(object):
    def __init__(self, order=None, brainfile=None, timeout=None, banwords=None, banwordfile=None, max_length=None):
        """
        Args:
            banwords (list of str, optional): List of common words (in
                that MegaHAL should ignore when learning and replying.
                Defaults to DEFAULT_BANWORDS. Pro tip: do a web search for the
                most commonly used words in your language and use the top ~300
                of those.
        """
        self.max_length = max_length
        if banwordfile and not banwords:
            banwords = []
            with open(banwordfile, "r") as f:
                for line in f:
                    banwords.extend([w.strip().upper() for w in line.split(",")])
        elif banwords:
            banwords = [w.upper() for w in banwords]
        self.__brain = Brain(order, brainfile, timeout=timeout, banwords=banwords)

    @property
    def brainsize(self):
        """Subtract by 2 because dictionary contains <ERROR> & <FIN> symbols"""
        return len(self.__brain.dictionary) - 2

    @property
    def banwords(self):
        """This is a list of words which cannot be used as keywords"""
        return self.__brain.banwords

    @property
    def auxwords(self):
        """This is a list of words which can be used as keywords only in order to supplement other keywords"""
        return self.__brain.auxwords

    @property
    def swapwords(self):
        """The word on the left is changed to the word on the right when used as a keyword"""
        return self.__brain.swapwords

    def train(self, file):
        """Train the brain with textfile, each line is a phrase"""
        line_count = 0
        percent = 0
        line_number = 1
        with open(file, 'r') as fp:
            for line in fp:
                line_count += 1
        with open(file, 'r') as fp:
            for line in fp:
                if int((line_number / line_count) * 100) != percent:
                    percent = int((line_number / line_count) * 100)
                    print('{} %'.format(percent))
                # Remove quotation marks
                line = line.replace('"', '')
                # Remove superfluous spaces
                line = re.sub(r'\s{2,}', ' ', line)
                # Remove whitespace at beginning and end
                line = line.strip()
                # Remove dash (and following spaces) in beginning
                line = re.sub(r'^-\s*', '', line)
                # Exclude empty lines, comment lines, and lines without
                # "word" characters. Also some special rules
                if line \
                        and not line.startswith('#') \
                        and re.search(r'\w', line) \
                        and "kapitlet" not in line.lower() \
                        and "psaltaren" not in line.lower():
                    self.learn(line)
                line_number += 1

    def learn(self, phrase):
        """Learn from phrase"""
        self.__brain.communicate(phrase, reply=False)

    def get_reply(self, phrase, max_length=None, timeout=None):
        """Get a reply based on the phrase"""
        return self.__brain.communicate(phrase, max_length=max_length or self.max_length, timeout=timeout)

    def get_reply_nolearn(self, phrase, max_length=None, timeout=None):
        """Get a reply without updating the database"""
        return self.__brain.communicate(phrase, learn=False, max_length=max_length or self.max_length, timeout=timeout)

    def interact(self, timeout=None):
        """Have a friendly chat session.. ^D to exit"""
        print("Go ahead and chat! ^D to exit.")
        while True:
            try:
                phrase = input('>>> ')
            except EOFError:
                break
            if phrase:
                print(self.get_reply(phrase, timeout=timeout or 2))

    def sync(self):
        """Flush any changes to disk"""
        self.__brain.sync()

    def close(self):
        """Close database"""
        self.__brain.close()

    def clear(self):
        self.__brain.init_db(clear=True)
