import os
import re
import string
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

            new_sentence = Sentence(tokens, int(sentence.attrib['n']))
            self.sentences.append(new_sentence)

    def add_text_file_to_corpus(self, file_name: str):
        """
        This method will receive a file name, such that the file is a text file (from Wikipedia), read the content
        from it and add it to the corpus.
        :param: file_name: The name of the text file that will be read
        :return: None
        """
        # read the file and create raw sentences
        wiki_file = open(file_name, 'r')
        wiki_txt = wiki_file.read()
        enhanced_txt = self.adjust_txt_with_edge_cases(wiki_txt)
        raw_sentences = self.create_sentences(enhanced_txt)

        # Get the latest sentence index
        sentence_index = 1 if len(self.sentences) == 0 else self.sentences[-1].index + 1
        for sentence in raw_sentences:
            enhanced_sentence = self.enhance_sentence(sentence[0])
            # get non-empty words
            words = list(filter(lambda token: token != '', enhanced_sentence.split(' ')))
            tokens_list = []
            # create tokens
            for word in list(dict.fromkeys(words)):  # unique words
                is_word = False if word in string.punctuation else True
                new_token = Token(
                    value=word,
                    token_type='w' if is_word else 'c',
                )
                tokens_list.append(new_token)

            # Add the end mark
            if sentence[1] != '\n':
                tokens_list.append(Token('c', sentence[1]))
            new_sentence = Sentence(tokens_list, sentence_index)
            sentence_index += 1
            self.sentences.append(new_sentence)

        wiki_file.close()

    @staticmethod
    def adjust_txt_with_edge_cases(txt):
        """
        This method takes the text and adjusts it before splitting it to sentences.
        The main assumption is that sentences are being separated by dots, so whenever we find a "non-end-of-sentence"
        dot, we'll replace it with "##" and after the split, return it back to a dot.
        It does the following adjustments:
        1. Decimals: 3.4 -> 3##4
        2. Abbreviations: Sr. / U.S. -> Sr## / U##S##
        3. Trailing points: "And so on..." -> "And so on######"
        4. Special cases: (see below)
        5. Headers: removes ==
        :param txt: the text to adjust
        :return: the adjusted text
        """
        words_to_adjust = []
        regexes = [
            r'\d+\.\d+',  # 3.4 / 5.77
            r'\s[A-Z]?[a-z]\.',  # Sr. / v. / Lt.
            r'\s(?:[A-Z]\.)+',  # U.S. / M.D.
            r'(?:[a-z]\.){2,}',  # e.g. / i.e.
            r'\.{2,}'  # "And so on..." -> "And so on######"
        ]
        for regex in regexes:
            words_to_adjust = words_to_adjust + re.findall(regex, txt)

        # need to sort the expressions by their length in a desc order in order for "U.S." to be replaced before "S."
        words_to_adjust = sorted(set(words_to_adjust), key=len, reverse=True)

        # Adjust found words
        for word in words_to_adjust:
            txt = txt.replace(word, str(word).replace('.', '##'))

        special_cases = {
            '.com': '##com',
            'Fed.': 'Fed##',
            '. ,': '# ,',
            '."': '##"',
            '(listen)': ''
        }
        for word, replaced_txt in special_cases.items():
            txt = txt.replace(word, replaced_txt)

        special_cases_with_regex = {
            r'(=)+': ' ',  # == / === etc
        }
        for pattern, replaced_txt in special_cases_with_regex.items():
            txt = re.sub(pattern, replaced_txt, txt)

        return txt

    @staticmethod
    def create_sentences(txt):
        txt = txt + '\n'  # need to add this for the end_marks to be the same size of raw_sentences
        # Split to sentences
        raw_sentences = re.split(r'[.;\n]', txt)
        # Save the ending mark of each sentence
        end_marks = re.findall(r'[.;\n]', txt)
        enhanced = []
        # create list of (sentence, end_mark)
        for count, raw_sent in enumerate(raw_sentences):
            adjusted_sentence = raw_sent.replace('##', '.').strip()
            if adjusted_sentence != '':
                enhanced.append((adjusted_sentence, end_marks[count]))
        return enhanced

    @staticmethod
    def enhance_sentence(sentence):
        # before splitting into tokens, add " " before and after punctuations
        for sign in string.punctuation:
            sentence = sentence.replace(sign, ' ' + sign + ' ')
        return sentence

    def create_text_file(self, file_name: str):
        """
        This method will write the content of the corpus in the manner explained in the exercise instructions.
        :param: file_name: The name of the file that the text will be written on
        :return: None
        """
        file_str = ''
        for sentence in self.sentences:
            for token in sentence.tokens:
                file_str += token.value + ' '
            file_str += '\n'
        output_file = open(file_name, 'w', encoding='utf8')
        output_file.write(file_str)


# Implement an n-gram language model class, that will be built using a corpus of type "Corpus" (thus, you will need to
# connect it in any way you want to the "Corpus" class):


class NGramModel:

    def __init__(self):
        return


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
