from __future__ import division

import sys
import time

import numpy as np


def parse_babi_task(data_files, dictionary, given_max_words=np.Inf, given_max_sentences=np.Inf, include_question=False):
    """ Parse bAbI data.

    Args:
       data_files (list): a list of data file's relative paths.
       dictionary (dict): word's dictionary, (key: value) = (word: number)
       include_question (bool): whether count question toward input sentence.

    Returns:
        A tuple of (story, questions, qstory):
            story (3-D array)
                [position of word in sentence, sentence index, story index] = index of word in dictionary
            questions (2-D array)
                [0-9, question index], in which the first component is encoded as follows:
                    0 - story index
                    1 - index of the last sentence before the question
                    2 - index of the answer word in dictionary
                    3 to 13 - indices of supporting sentence
                    14 - line index
            qstory (2-D array) question's indices within a story
                [index of word in question, question index] = index of word in dictionary
    """
    # Try to reserve spaces beforehand (large matrices for both 1k and 10k data sets)
    # maximum number of words in sentence = 20
    #
    # For each task, we have 1000 or 10k questions for training
    story = np.zeros((20, 500, len(data_files) * 3500), np.int16)
    questions = np.zeros((14, len(data_files) * 10000), np.int16)
    qstory = np.zeros((20, len(data_files) * 10000), np.int16)

    # NOTE: question's indices are not reset when going through a new story
    #       max_words     : maximum number of words in a sentence
    #       max_sentences : maximum number of sentences in a story
    #       question_idx  : indice among all the 1000/10000 questions
    #       sentence_idx  : indice among a certain story
    story_idx, question_idx, sentence_idx, max_words, max_sentences = -1, -1, -1, 0, 0

    # Mapping line number (within a story) to sentence's index (to support the flag include_question)
    # When questions are not included, this term will be useful
    mapping = None

    for fp in data_files:
        with open(fp) as f:
            for line_idx, line in enumerate(f):
                line = line.rstrip().lower()
                words = line.split()

                # Story begins
                if words[0] == '1':
                    story_idx += 1
                    sentence_idx = -1
                    mapping = []

                # FIXME: This condition makes the code more fragile!
                if '?' not in line:
                    is_question = False
                    sentence_idx += 1
                else:
                    is_question = True
                    question_idx += 1
                    questions[0, question_idx] = story_idx
                    questions[1, question_idx] = sentence_idx
                    if include_question:
                        sentence_idx += 1

                mapping.append(sentence_idx)

                # Skip substory index
                for k in range(1, len(words)):
                    w = words[k]

                    # remove useless punctuation
                    if w.endswith('.') or w.endswith('?'):
                        w = w[:-1]

                    if w not in dictionary:
                        dictionary[w] = len(dictionary)

                    if max_words < k:
                        max_words = k

                    if not is_question:
                        story[k - 1, sentence_idx, story_idx] = dictionary[w]
                    else:
                        qstory[k - 1, question_idx] = dictionary[w]
                        if include_question:
                            story[k - 1, sentence_idx, story_idx] = dictionary[w]

                        # NOTE: Punctuation is already removed from w
                        #
                        # Since '?' has been removed from w, we can only
                        # use words[k] to test whether there is '?'
                        #
                        # Word after question is the answer
                        if words[k].endswith('?'):
                            answer = words[k + 1]
                            if answer not in dictionary:
                                dictionary[answer] = len(dictionary)

                            questions[2, question_idx] = dictionary[answer]

                            # Indices of supporting sentences
                            for h in range(k + 2, len(words)):
                                questions[1 + h - k, question_idx] = mapping[int(words[h]) - 1]

                            questions[-1, question_idx] = line_idx
                            break

                if max_sentences < sentence_idx + 1:
                    max_sentences = sentence_idx + 1

    story = story[:min(max_words, given_max_words), :min(max_sentences, given_max_sentences), :(story_idx + 1)]
    questions = questions[:, :(question_idx + 1)]
    qstory = qstory[:min(max_words, given_max_words), :(question_idx + 1)]

    return story, questions, qstory


class Progress(object):
    """
    Progress bar
    """

    def __init__(self, iterable, bar_length=50):
        self.iterable = iterable
        self.bar_length = bar_length
        self.total_length = len(iterable)
        self.start_time = time.time()
        self.count = 0

    def __iter__(self):
        for obj in self.iterable:
            yield obj
            self.count += 1
            percent = self.count / self.total_length
            print_length = int(percent * self.bar_length)
            progress = "=" * print_length + " " * (self.bar_length - print_length)
            elapsed_time = time.time() - self.start_time
            print_msg = "\r|%s| %.0f%% %.1fs" % (progress, percent * 100, elapsed_time)
            sys.stdout.write(print_msg)
            if self.count == self.total_length:
                sys.stdout.write("\r" + " " * len(print_msg) + "\r")
            sys.stdout.flush()
