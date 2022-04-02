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
            # Adding sentence start token at the beginning of the sentence
            tokens = [Token('s', Token.SENTENCE_START_TOK)]
            # tokens = []
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
                    self.num_of_words += 1

            # Adding sentence end token at the end of the sentence
            tokens.append(Token('s', Token.SENTENCE_END_TOK))
            new_sentence = Sentence(tokens, int(sentence.attrib['n']))
            self.sentences.append(new_sentence)
            # Saving the sentence length. Will be used in the random sentence generation
            self.sentences_lengths.append(len(sentence))

    def get_tokens(self):
        # get a list of all tokens in their lower case form
        tokens_list = []
        for sen in self.sentences:
            tokens_list.extend([tok.value.lower() for tok in sen.tokens])
        return tokens_list


class NGramModel:
    def __init__(self, max_n, corpus, linear_interpolation_params: tuple):
        """
        The NGramModel object holds several models' calculation and is defined by the max_n param
        i.e: If max_n = 3, the obj will be able to perform as unigarm, Bigram and Trigram models.
        All (public) methods accept the n that determines on which model the method will work
        """
        self.linear_interpolation_params = linear_interpolation_params
        self.corpus = corpus
        self.max_n = max_n
        self.voc_sizes = []

        # Create unigram model
        tokens = self.corpus.get_tokens()
        self.voc_sizes.append(len(set(tokens)))
        self.num_of_words = len(
            [tok for tok in tokens if Token.SENTENCE_START_TOK not in tok and Token.SENTENCE_END_TOK not in tok])

        n_words_combinations = [tokens]

        # create Ngram model for k = 2 to max_n
        for k in range(2, max_n + 1):
            k_combinations = []
            for i in range(len(tokens) - k + 1):
                k_combinations.append(' '.join(tokens[i:i + k]))
            n_words_combinations.append(k_combinations)
            self.voc_sizes.append(len(set(
                [comb for comb in k_combinations
                 if Token.SENTENCE_START_TOK not in comb and Token.SENTENCE_END_TOK not in comb]
            )))

        counters = [dict(Counter(combination)) for combination in n_words_combinations]
        self.n_tokens_counters = counters

    def calculate_sentence_probability(self, sentence, n):
        tokens = self._tokenize_sentence(sentence)
        prob = 0
        # calculate i < n grams for the beginning of the sentence
        for i in range(n - 1):
            tok = ' '.join(tokens[:i + 1])
            prob += self._calculate_combination_probability(tok)
        # calculate n grams
        for i in range(n - 1, len(tokens)):
            tok = ' '.join(tokens[i - n + 1:i + 1])
            prob += self._calculate_combination_probability(tok)
        return prob

    @staticmethod
    def _tokenize_sentence(sentence):
        punctuations = r"""?!().,‘:;[]{}|"""
        for sign in punctuations:
            sentence = sentence.replace(sign, ' ' + sign + ' ')

        sentence = sentence.replace('\'', ' \'')
        sentence = sentence.replace('n \'t', ' n\'t')

        tokens = [tok for tok in sentence.split(' ') if tok != '']
        return tokens

    def _calculate_combination_probability(self, combination):
        combination_tokens = combination.split(' ')
        if len(combination_tokens) != 3:
            return self._calculate_probability_with_laplace(combination)
        else:
            # linear interpolation
            uni = self._calculate_probability_with_laplace(combination_tokens[-1])
            bi = self._calculate_probability_with_laplace(' '.join(combination_tokens[1:]))
            tri = self._calculate_probability_with_laplace(combination)
            return (uni * self.linear_interpolation_params[0]) + (bi * self.linear_interpolation_params[1]) + (
                    tri * self.linear_interpolation_params[2])

    def _calculate_probability_with_laplace(self, combination: str):
        combination_parts = combination.split(' ')
        combination_len = len(combination_parts)
        # calculate the first part of the denominator
        num_of_w = self.num_of_words if combination_len == 1 else self.n_tokens_counters[combination_len - 1].get(
            combination_parts[0], 0)

        num = (self.n_tokens_counters[combination_len - 1].get(combination, 0) + 1) / (
                num_of_w + self.voc_sizes[combination_len - 1])
        return math.log(num)

    def generate_random_sentence(self, n):
        # generate a random length
        max_sentence_length = random.choice(self.corpus.sentences_lengths)
        sen = [Token.SENTENCE_START_TOK]
        sentence_length = 1

        population = list(self.n_tokens_counters[n - 1].keys())
        while sentence_length < max_sentence_length and sen[-1] != Token.SENTENCE_END_TOK:
            if n == 1:
                picked_token = random.choice(population)
            else:
                # The first prediction in trigram need to be drawn from the bigram dict
                if n == 3 and sentence_length == 1:
                    curr_population = [token for token in list(self.n_tokens_counters[1].keys()) if
                                       token.startswith(Token.SENTENCE_START_TOK)]
                    curr_weights = [self.n_tokens_counters[1][token] for token in curr_population]
                else:
                    curr_population = [token for token in population if token.startswith(' '.join(sen[1 - n:]) + ' ')]
                    curr_weights = [self.n_tokens_counters[n - 1][token] for token in curr_population]
                picked_combination = random.choices(curr_population, curr_weights)[0]
                picked_token = picked_combination.split(' ')[-1]

            sen.append(picked_token)
            sentence_length += 1

        # remove start/end of sentence tokens
        sen = sen[1:] if sen[-1] != Token.SENTENCE_END_TOK else sen[1:len(sen) - 1]

        return ' '.join(sen)


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

    print('XML files Additions - In Progress...')
    xml_files_names = os.listdir(xml_dir)  # [:3]
    for file in xml_files_names:
        corpus.add_xml_file_to_corpus(os.path.join(xml_dir, file))
    print('XML files Additions - Done!')

    # 2. Create a language model based on the corpus.
    print('Model Building - In Progress...')
    max_n = 3
    linear_interpolation_params = (0.2, 0.35, 0.45)
    model = NGramModel(max_n, corpus, linear_interpolation_params)
    models_names = ('Unigrams Model', 'Bigrams Model', 'Trigrams Model')
    print('Model Building - Done!')

    # 3. Calculate and print onto the output file the first task, in the wanted format.
    print('Sentences Probability Calculation - In Progress...')
    sentences = [
        'May the Force be with you.',
        'I\'m going to make him an offer he can\'t refuse.',
        'Ogres are like onions.',
        'You\'re tearing me apart, Lisa!',
        'I live my life one quarter at a time.'
    ]
    output_str = '*** Sentence Predictions ***\n\n'
    for n, model_name in enumerate(models_names):
        output_str += model_name + ':\n\n'
        for sentence in sentences:
            prob = model.calculate_sentence_probability(sentence.lower(), n + 1)
            output_str += sentence + '\nProbability: ' + str(prob) + '\n'
        output_str += '\n'
    print('Sentences Probability Calculation - Done!')

    # 4. Print onto the output file the results from the second task in the wanted format.
    print('Random Sentences Generation - In Progress...')
    num_of_sentences = 5
    output_str += '\n*** Random Sentence Generation ***\n\n'
    for n, model_name in enumerate(models_names):
        output_str += model_name + ':\n\n'
        for i in range(num_of_sentences):
            output_str += model.generate_random_sentence(n + 1) + '\n'
        output_str += '\n'
    print('Random Sentences Generation - Done!')

    print(f'Writing output to {output_file_path}')
    output_file = open(output_file_path, 'w', encoding='utf8')
    output_file.write(output_str)
    output_file.close()
    print(f'Program ended.')


if __name__ == '__main__':
    main()
