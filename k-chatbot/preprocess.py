# Import module
import os
import re

import numpy as np
import pandas as pd
from tqdm import tqdm

from konlpy.tag import Okt

# Set preprocessing values
'''
Token:
<PAD> : padding
<SOS> : start
<END> : end
<UNK> : word not in the dictionary
'''
FILTERS = "([~.,!?\"':;)(])"
PAD = "<PAD>"
STD = "<SOS>"
END = "<END>"
UNK = "<UNK>"

PAD_INDEX = 0
STD_INDEX = 1
END_INDEX = 2
UNK_INDEX = 3

MARKER = [PAD, STD, END, UNK]
CHANGE_FILTER = re.compile(FILTERS)

MAX_SEQUENCE = 25


# Load data
def load_data(path):
    data_df = pd.read_csv(path, header=0)
    query, answer = list(data_df['Q']), list(data_df['A'])

    return query, answer


# Text preprocessing and Create word list
def data_tokenizer(data):
    words = []
    for sentence in data:
        # Change special symbols(in FILTER) to ""
        sentence = re.sub(CHANGE_FILTER, "", sentence)
        # Split words by space
        for word in sentence.split():
            words.append(word)

    return [word for word in words if word]


# Separate text into morphemes
def prepro_like_morphlized(data):
    morph_analyzer = Okt()
    result_data = list()
    for seq in tqdm(data):
        morphlized_seq = " ".join(morph_analyzer.morphs(seq.replace(' ', '')))
        result_data.append(morphlized_seq)

    return result_data


# Make a word vocabulary
def load_vocabulary(path, vocab_path, tokenize_as_morph=False):
    vocabulary_list = []

    # If there is no word vocabulary in the path
    if not os.path.exists(vocab_path):
        if os.path.exists(path):
            data_df = pd.read_csv(path, encoding='utf-8')
            query, answer = list(data_df['Q']), list(data_df['A'])

            if tokenize_as_morph:
                query = prepro_like_morphlized(query)
                answer = prepro_like_morphlized(answer)

            data = []
            data.extend(query)
            data.extend(answer)

            words = data_tokenizer(data)
            # Delete duplicate words
            words = list(set(words))
            words[:0] = MARKER
        # Create a vocabulary file and put the words
        with open(vocab_path, 'w', encoding='utf-8') as vocabulary_file:
            for word in words:
                vocabulary_file.write(word + '\n')

    # If there is word vocabulary in the path
    with open(vocab_path, 'r', encoding='utf-8') as vocabulary_file:
        for line in vocabulary_file:
            vocabulary_list.append(line.strip())

    # char2idx = {word: index}, idx2char = {idx: word}
    char2idx, idx2char = make_vocabulary(vocabulary_list)

    return char2idx, idx2char, len(char2idx)


# Make a contents of a list into a dictionary
def make_vocabulary(vocabulary_list):
    # dictionary: {key: word, value: index}
    char2idx = {char: idx for idx, char in enumerate(vocabulary_list)}
    # dictionary: {key: index, value: word}
    idx2char = {idx: char for idx, char in enumerate(vocabulary_list)}

    return char2idx, idx2char


# Make the input of the encoder
def enc_processing(value, dictionary, tokenize_as_morph=False):
    sequences_input_index = []
    sequences_length = []

    if tokenize_as_morph:
        value = prepro_like_morphlized(value)

    for sequence in value:
        # Remove special characters in CHANGE_FILTER
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        sequence_index = []

        # Separate sentences by space
        for word in sequence.split():
            # If the word is in the word vocabulary
            if dictionary.get(word) is not None:
                sequence_index.extend([dictionary[word]])
            # If the word is not in the word vocabulary
            else:
                # Put UNK token
                sequence_index.extend([dictionary[UNK]])

        # If the sentence is longer than the MAX_SEQUENCE
        if len(sequences_length) > MAX_SEQUENCE:
            # Cut sentence to the MAX_SEQUENCE
            sequence_index = sequence_index[:MAX_SEQUENCE]
        # Append the length of sentence
        sequences_length.append(len(sequence_index))

        # If the sentence is shorter than the MAX_SEQUENCE
        # Add the PAD token in the empty part
        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]

        # Append the indexed value
        sequences_input_index.append(sequence_index)

    # Convert the indexed array to a numpy array and return with the length of statement.
    return np.asarray(sequences_input_index), sequences_length


# Make the input of the decoder
def dec_output_processing(value, dictionary, tokenize_as_morph=False):
    sequences_output_index = []
    sequences_length = []

    if tokenize_as_morph:
        value = prepro_like_morphlized(value)

    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        # Decoder input starts with STD token
        sequence_index = [dictionary[STD]] + [dictionary[word] if word in dictionary else dictionary[UNK] for word in sequence.split()]

        if len(sequence_index) > MAX_SEQUENCE:
            sequence_index = sequence_index[:MAX_SEQUENCE]
        sequences_length.append(len(sequence_index))

        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]
        sequences_output_index.append(sequence_index)

    return np.asarray(sequences_output_index), sequences_length


# Make a target value
def dec_target_processing(value, dictionary, tokenize_as_morph=False):
    sequences_target_index = []

    if tokenize_as_morph:
        value = prepro_like_morphlized(value)

    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        sequence_index = [dictionary[word] if word in dictionary else dictionary[UNK] for word in sequence.split()]
        if len(sequence_index) >= MAX_SEQUENCE:
            # Target ends with END token
            sequence_index = sequence_index[:MAX_SEQUENCE - 1] + [dictionary[END]]
        else:
            sequence_index += [dictionary[END]]

        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]
        sequences_target_index.append(sequence_index)

    return np.asarray(sequences_target_index)
