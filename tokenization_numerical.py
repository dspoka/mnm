# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""

from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import logging
import os
import sys
import unicodedata
from io import open
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {'vocab_file': 'vocab.txt'}

PRETRAINED_VOCAB_FILES_MAP = {
    'vocab_file':
    {
        'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt",
        'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt",
        'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt",
        'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt",
        'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-vocab.txt",
        'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt",
        'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt",
        'bert-base-german-cased': "https://int-deepset-models-bert.s3.eu-central-1.amazonaws.com/pytorch/bert-base-german-cased-vocab.txt",
        'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-vocab.txt",
        'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-vocab.txt",
        'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-vocab.txt",
        'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-vocab.txt",
        'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-vocab.txt",
    }
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    'bert-base-uncased': 512,
    'bert-large-uncased': 512,
    'bert-base-cased': 512,
    'bert-large-cased': 512,
    'bert-base-multilingual-uncased': 512,
    'bert-base-multilingual-cased': 512,
    'bert-base-chinese': 512,
    'bert-base-german-cased': 512,
    'bert-large-uncased-whole-word-masking': 512,
    'bert-large-cased-whole-word-masking': 512,
    'bert-large-uncased-whole-word-masking-finetuned-squad': 512,
    'bert-large-cased-whole-word-masking-finetuned-squad': 512,
    'bert-base-cased-finetuned-mrpc': 512,
}

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip('\n')
        vocab[token] = index
    return vocab


def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens


class BertNumericalTokenizer(PreTrainedTokenizer):
    r"""
    Constructs a BertTokenizer.
    :class:`~pytorch_transformers.BertTokenizer` runs end-to-end tokenization: punctuation splitting + wordpiece

    Args:
        vocab_file: Path to a one-wordpiece-per-line vocabulary file
        do_lower_case: Whether to lower case the input. Only has an effect when do_wordpiece_only=False
        do_basic_tokenize: Whether to do basic tokenization before wordpiece.
        max_len: An artificial maximum length to truncate tokenized sequences to; Effective maximum length is always the
            minimum of this value (if specified) and the underlying BERT model's sequence length.
        never_split: List of tokens which will never be split during tokenization. Only has an effect when
            do_wordpiece_only=False
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, vocab_file, do_lower_case=True, do_basic_tokenize=True, never_split=None,
                 unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]",
                 mask_token="[MASK]", tokenize_chinese_chars=True, **kwargs):
        """Constructs a BertNumericalTokenizer.

        Args:
            **vocab_file**: Path to a one-wordpiece-per-line vocabulary file
            **do_lower_case**: (`optional`) boolean (default True)
                Whether to lower case the input
                Only has an effect when do_basic_tokenize=True
            **do_basic_tokenize**: (`optional`) boolean (default True)
                Whether to do basic tokenization before wordpiece.
            **never_split**: (`optional`) list of string
                List of tokens which will never be split during tokenization.
                Only has an effect when do_basic_tokenize=True
            **tokenize_chinese_chars**: (`optional`) boolean (default True)
                Whether to tokenize Chinese characters.
                This should likely be deactivated for Japanese:
                see: https://github.com/huggingface/pytorch-pretrained-BERT/issues/328
        """
        super(BertNumericalTokenizer, self).__init__(unk_token=unk_token, sep_token=sep_token,
                                            pad_token=pad_token, cls_token=cls_token,
                                            mask_token=mask_token, **kwargs)


        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertNumericalTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file))
        self.vocab = load_vocab(vocab_file)

        self.unk_num = '[UNK_NUM]'
        
        self.default_value = 1.0

        never_split = ['[UNK_NUM]']

        
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])

        self.do_basic_tokenize = do_basic_tokenize

        self.numerical_tokenizer = NumericalTokenizer(do_lower_case=do_lower_case,
                                                  never_split=never_split)


        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
                                                  never_split=never_split,
                                                  tokenize_chinese_chars=tokenize_chinese_chars)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token, unk_num=self.unk_num)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _tokenize(self, text, get_values=False, get_sigfigs=None, get_numeric_masks=None):
        split_tokens = []
        numeric_values = []
        numeric_masks = []
        split_sigfigs = []
        i = 0
        for (token, sigfig) in self.numerical_tokenizer.tokenize(text, never_split=self.all_special_tokens):
            for (sub_token, numeric_value, numeric_mask) in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
                numeric_values.append(numeric_value)
                numeric_masks.append(numeric_mask)

                
                if numeric_value != self.default_value:
                    split_sigfigs.append(sigfig)
                else:
                    split_sigfigs.append('-1')


                if numeric_value != self.default_value and sub_token != self.unk_num:
                    print(sub_token, numeric_value)
                    foohere

        if get_numeric_masks:
            return numeric_masks

        if get_values:
            return numeric_values

        assert len(split_tokens) == len(numeric_values) == len(split_sigfigs)
        if get_sigfigs:
            return split_sigfigs

        return split_tokens

    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """

        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    def convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = ' '.join(tokens).replace(' ##', '').strip()
        return out_string

    def save_vocabulary(self, vocab_path):
        """Save the tokenizer vocabulary to a directory or file."""
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, VOCAB_FILES_NAMES['vocab_file'])
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning("Saving vocabulary to {}: vocabulary indices are not consecutive."
                                   " Please check that the vocabulary is not corrupted!".format(vocab_file))
                    index = token_index
                writer.write(token + u'\n')
                index += 1
        return (vocab_file,)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """ Instantiate a BertNumericalTokenizer from pre-trained vocabulary files.
        """
        if pretrained_model_name_or_path in PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES:
            if '-cased' in pretrained_model_name_or_path and kwargs.get('do_lower_case', True):
                logger.warning("The pre-trained model you are loading is a cased model but you have not set "
                               "`do_lower_case` to False. We are setting `do_lower_case=False` for you but "
                               "you may want to check this behavior.")
                kwargs['do_lower_case'] = False
            elif '-cased' not in pretrained_model_name_or_path and not kwargs.get('do_lower_case', True):
                logger.warning("The pre-trained model you are loading is an uncased model but you have set "
                               "`do_lower_case` to False. We are setting `do_lower_case=True` for you "
                               "but you may want to check this behavior.")
                kwargs['do_lower_case'] = True

        return super(BertNumericalTokenizer, cls)._from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)


class NumericalTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True, never_split=None):
        """ Constructs a BasicTokenizer.

        Args:
            **do_lower_case**: Whether to lower case the input.
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes.
                Now implemented directly at the base class level (see :func:`PreTrainedTokenizer.tokenize`)
                List of token not to split.
        """
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def tokenize(self, text, never_split=None):
        """ Basic Numerical Tokenization of a piece of text.

        Args:
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes.
                Now implemented directly at the base class level (see :func:`PreTrainedTokenizer.tokenize`)
                List of token not to split.
        """
        # digits = '0123456789'
        # punctuation = '$%'

        # text = self._clean_text(text)
        # orig_tokens = whitespace_tokenize(text)
        split_tokens, split_sigfigs = normalize_numbers_in_sent(text)

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        output_sigfigs = whitespace_tokenize(" ".join(split_sigfigs))
        return zip(output_tokens,split_sigfigs)
        # return output_tokens,

# _numbers = '[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?'
# fraction_pattern = re.compile(_fraction)
# number_pattern = re.compile(_numbers)

class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True, never_split=None, tokenize_chinese_chars=True):
        """ Constructs a BasicTokenizer.

        Args:
            **do_lower_case**: Whether to lower case the input.
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes.
                Now implemented directly at the base class level (see :func:`PreTrainedTokenizer.tokenize`)
                List of token not to split.
            **tokenize_chinese_chars**: (`optional`) boolean (default True)
                Whether to tokenize Chinese characters.
                This should likely be deactivated for Japanese:
                see: https://github.com/huggingface/pytorch-pretrained-BERT/issues/328
        """
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = never_split
        self.tokenize_chinese_chars = tokenize_chinese_chars

    def tokenize(self, text, never_split=None):
        """ Basic Tokenization of a piece of text.
            Split on "white spaces" only, for sub-word tokenization, see WordPieceTokenizer.

        Args:
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes.
                Now implemented directly at the base class level (see :func:`PreTrainedTokenizer.tokenize`)
                List of token not to split.
        """
        never_split = self.never_split + (never_split if never_split is not None else [])
        text = self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text, never_split=None):
        """Splits punctuation on a piece of text."""
        if never_split is not None and text in never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            #dont split on periods if number is before it
            # if _is_punctuation(char) and not chars[i-1].isdigit() or _is_punctuation(char) and i == 0:
            if _is_punctuation(char):
                if i == 0:
                    do_split = True
                elif i == len(chars)-1:
                    do_split = True
                else:
                    if not chars[i-1].isdigit():
                        do_split = True
                    else:
                        do_split = False
            else:
                do_split = False

            if do_split:
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)


class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, unk_num, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.unk_num = unk_num
        self.default_value = 1.0
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        numeric_values = []
        numeric_mask = []

        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                numeric_values.append(self.default_value)
                numeric_mask.append(0)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                try:
                    if token not in ['infinity', 'inf', 'nan']:
                        numeric_value = float(token)
                        is_number = True
                    else:
                        is_number = False
                except:
                    ValueError
                    is_number = False

                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab and is_number == False:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_number:
                #ACTUAL NUMBER HERE
                output_tokens.append(self.unk_num)
                numeric_values.append(numeric_value)
                numeric_mask.append(1)
            elif is_bad:
                output_tokens.append(self.unk_token)
                numeric_values.append(self.default_value)#-9e9
                numeric_mask.append(0)
            else:
                numeric_values.extend([self.default_value]*len(sub_tokens))#-9e9
                numeric_mask.extend([0]*len(sub_tokens))
                output_tokens.extend(sub_tokens)
        assert len(numeric_values) == len(output_tokens) == len(numeric_mask)

        return zip(output_tokens, numeric_values, numeric_mask)


def _is_whitespace(char):
    """Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def _is_control(char):
    """Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    # if cat.startswith("P") and cp != 46:
    if cat.startswith("P"):
        return True
    return False
################
#

Small = {
    'zero': 0.0,
    'one': 1.0,
    'two': 2.0,
    'three': 3.0,
    'four': 4.0,
    'five': 5.0,
    'six': 6.0,
    'seven': 7.0,
    'eight': 8.0,
    'nine': 9.0,
    'ten': 10.0,
    'eleven': 11.0,
    'twelve': 12.0,
    'thirteen': 13.0,
    'fourteen': 14.0,
    'fifteen': 15.0,
    'sixteen': 16.0,
    'seventeen': 17.0,
    'eighteen': 18.0,
    'nineteen': 19.0,
    'twenty': 20.0,
    'thirty': 30.0,
    'forty': 40.0,
    'fifty': 50.0,
    'sixty': 60.0,
    'seventy': 70.0,
    'eighty': 80.0,
    'ninety': 90.0
}


Magnitude = {
    'thousand':     1000.0,
    'million':      1000000.0,
    'billion':      1000000000.0,
    'trillion':     1000000000000.0,
    'quadrillion':  1000000000000000.0,
    'quintillion':  1000000000000000000.0,
    'sextillion':   1000000000000000000000.0,
    'septillion':   1000000000000000000000000.0,
    'octillion':    1000000000000000000000000000.0,
    'nonillion':    1000000000000000000000000000000.0,
}

class NumberException(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)

def text2num(sent):

    if type(sent) is str:
        words = [word.lower() for word in sent.strip().split()]
    elif type(sent) is list:
        words = [word.lower() for word in sent]
    # n = 0
    # g = 0
    mantissa = 0
    # number = 0.0
    for i, word in enumerate(words):
        if i == 0:
            mantissa = Small.get(word, None)
            if mantissa is None:
                try:
                    mantissa = float(word)
                except ValueError:
                    raise NumberException("First must be a number of sorts")
        elif i != 0:
            magnitude = Magnitude.get(word, None)
            if magnitude is not None:
                mantissa = mantissa*magnitude
            else:  # non-number word
                raise NumberException("Unknown number: "+word)

    return mantissa

def generate_ngrams(sentence, n):
    return zip(*[sentence[i:] for i in range(n)])

def check_int(s):
    if s[0] in ('-', '+'):
        return s[1:].isdigit()
    return s.isdigit()

def preprocess(sent, remove_pos=False, never_split=None):
    """
    Preprocess the sentence by:
    . remove commas from numbers (2,000 -> 2000)
    . remove endings from ordinal numbers (2nd -> 2)
    . convert "a {hundred,thousand...}" to "one {hundred,thousand,...}" so it can be handled by text2num function
    . convert "digit digitword" (24 hundred) -> 2400
    and return the sentence's preprocessed list of words that should be passed into text2num.
    """
    if remove_pos:
        words = [word[:word.rfind('_')] for word in sent.strip().split()]
    else:
        words = [word for word in sent.strip().split()]
    tokenizer = BasicTokenizer(do_lower_case=True, never_split=never_split)
    words = tokenizer.tokenize(sent)
    # sent = ' '.join(tokens)

    words_lower = [word.lower() for word in words]
    # remove commas from numbers "2,000" -> 2000 and remove endings from ordinal numbers
    for i in range(len(words)):
        new_word = words_lower[i].replace(',', '')
        if new_word.endswith(('th', 'rd', 'st', 'nd')):
            new_word = new_word[:-2]
        try:
            if new_word not in ['infinity', 'inf', 'nan']:
                int_word = float(new_word)
                # words[i] = str(int_word)
                words[i] = new_word
        except ValueError:
            pass  # only modify this word if it's an int after preprocessing
    Magnitude_with_hundred = Magnitude.copy()
    Magnitude_with_hundred['hundred'] = 100
    # convert "a {hundred,thousand,million,...}" to "one {hundred,thousand,million,...}"
    for i in range(len(words)-1):
        if words_lower[i] == 'a' and words_lower[i+1] in Magnitude_with_hundred:
            words[i] = 'one'
    # convert "24 {Magnitude}" -> 24000000000000 (mix of digits and words)
    new_words = []
    sigs = []
    i = 0
    while i < len(words)-1:
        if check_int(words_lower[i]) and words_lower[i+1] in Magnitude_with_hundred:
            new_words.append(str(float(words_lower[i]) * Magnitude_with_hundred[words_lower[i+1]]))
            sigs.append(f'{words_lower[i]} {words_lower[i+1]}')
            i += 1
        else:
            new_words.append(words[i])
            sigs.append('')
            if i == len(words) - 2:
                new_words.append(words[i+1])
                sigs.append('')
        i += 1
    return new_words, sigs
# ​
#
def normalize_numbers_in_sent(sent, remove_pos=False, never_split=None):
    """
    Given a sentence, perform preprocessing and normalize number words to digits.
    :param sent: sentence (str)
    :return: a list of normalized words from the sentence
    """
    out_words = []
    words, sigfigs = preprocess(sent, remove_pos, never_split)
    out_sigfigs = []
    i = 0
    while i < len(words):
        for j in range(len(words), i, -1):
            try:
                number = str(text2num(words[i:j]))

                if sigfigs[i] == '':
                    out_sigfigs.append(' '.join(words[i:j]))
                else:
                    out_sigfigs.append(sigfigs[i])


                out_words.append(number)
                i = j-1  # skip this sequence since we replaced it with a number
                break
            except NumberException:
                if j == i+1:
                    out_sigfigs.append('-1')
                    out_words.append(words[i])
        i += 1
    assert len(out_sigfigs) == len(out_words)

    return out_words, out_sigfigs