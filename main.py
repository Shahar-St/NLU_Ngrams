import os
import xml.etree.ElementTree as ET
from sys import argv


class Token:
    def __init__(self, token_type, value, c5=None, hw=None, pos=None):
        self.token_type = token_type  # w (word) or c (punctuation marks)
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
            tokens = []
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

            new_sentence = Sentence(tokens, int(sentence.attrib['n']))
            self.sentences.append(new_sentence)


class NGramModel:
    def __init__(self, n, corpus):
        self.corpus = corpus
        self.n = n


def main():
    print('Program started')
    # xml_dir = argv[1]  # directory containing xml files from the BNC corpus, full path
    xml_dir = os.path.join(os.getcwd(), 'XML_files')
    # output_file = argv[2]  # output file name, full path
    output_file = os.path.join(os.getcwd(), 'output.txt')

    # Implement here your program:
    # 1. Create a corpus from the file in the given directory.
    print('Initializing Corpus')
    c = Corpus()

    print('Adding XML files')
    xml_files_names = os.listdir(xml_dir)
    for file in xml_files_names:
        c.add_xml_file_to_corpus(os.path.join(xml_dir, file))

    # 2. Create a language model based on the corpus.
    # 3. Calculate and print onto the output file the first task, in the wanted format.
    # 4. Print onto the output file the results from the second task in the wanted format.

    print('Program finished')


if __name__ == '__main__':
    main()
