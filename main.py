import os
import xml.etree.ElementTree as ET
from sys import argv
from collections import Counter
import math
import random


class Token:
    SENTENCE_START_TOK = '<s>'
    SENTENCE_END_TOK = '</s>'

    def __init__(self, token_type, value, c5=None, hw=None, pos=None):
        self.token_type = token_type  # w (word), c (punctuation marks), s (beg/end of sentence)
        self.value = value
        self.c5 = c5  # The C5 tag
        self.hw = hw  # headword
        self.pos = pos  # part of speech, derived from the c5 tag


class Sentence:
    def __init__(self, tokens_array, index: int):
        self.tokens = tokens_array
        self.tokens_num = len(tokens_array)
        self.index = index  # starts with 1


class Corpus:
    def __init__(self):
        self.sentences = []
        self.num_of_words = 0
        self.sentences_lengths = []

    def add_xml_file_to_corpus(self, file_name: str):
        """
        This method will receive a file name, such that the file is an XML file (from the BNC), read the content from
        it and add it to the corpus.
        :param: file_name: The name of the XML file that will be read
        :return: None
        """
        tree = ET.parse(file_name)

        # iterate over all sentences in the file and extract the tokens
        for sentence in tree.iter(tag='s'):
            tokens = [Token('s', Token.SENTENCE_START_TOK)]
            for word in sentence:
                if word.tag in ('w', 'c'):
                    att = word.attrib
                    new_token = Token(
                        token_type=word.tag,
                        c5=att['c5'],
                        hw=att.get('hw'),
                        pos=att.get('pos'),
                        value=word.text.strip()
                    )
                    tokens.append(new_token)
                    tokens.append(Token('s', Token.SENTENCE_END_TOK))
                    self.num_of_words += 1

            new_sentence = Sentence(tokens, int(sentence.attrib['n']))
            self.sentences.append(new_sentence)
            self.sentences_lengths.append(len(sentence))

    def get_tokens(self):
        tokens_list = []
        for sen in self.sentences:
            tokens_list.extend([tok.value.lower() for tok in sen.tokens])
        return tokens_list


class NGramModel:
    def __init__(self, max_n, corpus, linear_interpolation_params: tuple = None):
        self.linear_interpolation_params = linear_interpolation_params
        self.corpus = corpus
        self.max_n = max_n

        tokens = self.corpus.get_tokens()
        self.voc_size = len(set(tokens))
        self.num_of_words = len(tokens)

        # build model
        n_words_combinations = [tokens]

        for k in range(2, max_n + 1):
            k_combinations = []
            for i in range(len(tokens) - k + 1):
                k_combinations.append(' '.join(tokens[i:i + k]))
            n_words_combinations.append(k_combinations)

        counters = [Counter(combination) for combination in n_words_combinations]
        self.n_tokens_counters = counters

    def calculate_sentence_probability(self, sentence, n):
        tokens = sentence.split(' ')
        prob = 0
        # calculate i < n grams for the beginning of the sentence
        for i in range(n - 1):
            prob += self.calculate_combination_probability(' '.join(tokens[:i + 1]))

        # calculate n grams
        for i in range(n - 1, len(tokens)):
            prob += self.calculate_combination_probability(' '.join(tokens[i - n + 1:i + 1]))

        return prob

    def calculate_combination_probability(self, combination):
        combination_tokens = combination.split(' ')
        combination_len = len(combination_tokens)
        if combination_len != 3:
            # laplace
            return self.calculate_probability_with_laplace(combination)
        else:
            # linear interpolation
            # todo does it needs to be smoothed as well? if so they can be pre-built
            uni = self.calculate_probability_with_laplace(combination_tokens[-1])
            bi = self.calculate_probability_with_laplace(' '.join(combination_tokens[1:]))
            tri = self.calculate_probability_with_laplace(combination)
            return (uni * self.linear_interpolation_params[0]) + (bi * self.linear_interpolation_params[1]) + (
                    tri * self.linear_interpolation_params[2])

    def calculate_probability_with_laplace(self, combination: str):
        combination_len = len(combination.split(' '))
        return math.log(
            (self.n_tokens_counters[combination_len - 1].get(combination, 0) + 1) / (self.num_of_words + self.voc_size))

    def generate_random_sentence(self, n):
        max_sentence_length = random.choice(self.corpus.sentences_lengths)
        sen = Token.SENTENCE_START_TOK
        last_picked_token = Token.SENTENCE_START_TOK
        sentence_length = 0
        while sentence_length <= max_sentence_length and last_picked_token != Token.SENTENCE_END_TOK:
            last_picked_token = 'r'

            sen += ' ' + last_picked_token
            sentence_length += 1

        return sen


def main():
    print('Program started')
    # xml_dir = argv[1]  # directory containing xml files from the BNC corpus, full path
    xml_dir = os.path.join(os.getcwd(), 'XML_files')
    # output_file_path = argv[2]  # output file name, full path
    output_file_path = os.path.join(os.getcwd(), 'output.txt')

    # Implement here your program:
    # 1. Create a corpus from the file in the given directory.
    print('Initializing Corpus')
    corpus = Corpus()

    print('Adding XML files')
    xml_files_names = os.listdir(xml_dir)[:3]
    for file in xml_files_names:
        corpus.add_xml_file_to_corpus(os.path.join(xml_dir, file))

    # 2. Create a language model based on the corpus.
    print('Building model')
    max_n = 3
    linear_interpolation_params = (1, 1, 1)
    model = NGramModel(max_n, corpus, linear_interpolation_params)
    models_names = ('Unigrams Model', 'Bigrams Model', 'Trigrams Model')

    # 3. Calculate and print onto the output file the first task, in the wanted format.
    sentences = [
        # todo predict with <s> or without?
        'May the Force be with you',
        'I’m going to make him an offer he can’t refuse.',
        'Ogres are like onions.',
        'You’re tearing me apart, Lisa!',
        'I live my life one quarter at a time.'
    ]
    output_str = '*** Sentence Predictions ***\n\n'
    for n, model_name in enumerate(models_names):
        output_str += model_name + ':\n\n'
        for sentence in sentences:
            prob = model.calculate_sentence_probability(sentence.lower(), n + 1)
            output_str += sentence + '\nProbability: ' + str(prob) + '\n'
        output_str += '\n'

    output_file = open(output_file_path, 'w', encoding='utf8')

    # 4. Print onto the output file the results from the second task in the wanted format.
    num_of_sentences = 5
    output_str += '\n*** Random Sentence Generation ***\n\n'
    for n, model_name in enumerate(models_names):
        output_str += model_name + ':\n\n'
        for i in range(num_of_sentences):
            output_str += model.generate_random_sentence(n) + '\n'
        output_str += '\n'

    output_file.write(output_str)
    output_file.close()
    print('Program finished')


if __name__ == '__main__':
    main()
