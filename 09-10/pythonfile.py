import nltk

# Download the Brown corpus and the universal tagset
nltk.download("brown")
nltk.download("universal_tagset")
# Get the tagged sentences from the Brown corpus
tagged_sentences = list(nltk.corpus.brown.tagged_sents(tagset="universal"))
# The words converted to lower format
tagged_sentences_lower = [
    [(word.lower(), tag) for word, tag in sentence] for sentence in tagged_sentences
]

# The parts of speech
pos = {
    "ADJ": 0,
    "ADP": 1,
    "ADV": 2,
    "CONJ": 3,
    "DET": 4,
    "NOUN": 5,
    "NUM": 6,
    "PRON": 7,
    "PRT": 8,
    "VERB": 9,
    ".": 10,
    "X": 11,
    "START": 12,
    "END": 13,
}
pos_keys = list(pos.keys())
num_pos = 12
num_tags = 14
# The word map
word_map = {}
num_words = 0
for sentence in tagged_sentences_lower:
    for word, _ in sentence:
        if word not in word_map:
            word_map[word] = num_words
            num_words += 1

import numpy as np


def get_transition_matrix(dataset):
    transition = np.ones((num_tags, num_tags))

    for sentence in dataset:
        previous_tag = "START"
        for _, tag in sentence:
            transition[pos[previous_tag], pos[tag]] += 1
            previous_tag = tag
        transition[pos[previous_tag], pos["END"]] += 1  # Not necessary

    return transition / transition.sum(axis=1)


def get_emission_matrix(dataset):
    emission = np.ones((num_pos, num_words + 1))

    for sentence in dataset:
        for word, tag in sentence:
            emission[pos[tag], word_map[word]] += 1

    return emission / emission.sum(axis=1, keepdims=True)


class Dataset:
    def __init__(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set
        self.transition_matrix = get_transition_matrix(train_set)
        self.emission_matrix = get_emission_matrix(train_set)


datasets = []
n = len(tagged_sentences_lower)
sz = n // 5
parts = []
for i in range(5):
    parts.append(tagged_sentences_lower[i * sz : i * sz + sz])

test_set_arr = []

for i in range(5):
    train_set_arr = []
    for j in range(5):
        if j == i:
            continue
        train_set_arr.extend(parts[j])

    for sentence in parts[i]:
        sentence_string = ""
        for word, _ in sentence:
            sentence_string += word + " "
        test_set_arr.append(sentence_string.strip())

    datasets.append(Dataset(train_set_arr, test_set_arr))


def viterbi(transition, emission, sentence):

    parsed_sentence = sentence.lower().split()
    sentence_len = len(parsed_sentence)

    log_prob = np.full((sentence_len, num_pos), 0)
    prev = np.full((sentence_len, num_pos), -1)

    # Determining the probabilities:
    for state in range(num_pos):
        log_prob[0, state] += np.log(transition[pos["START"], state])
        if parsed_sentence[0] in word_map:
            log_prob[0, state] += np.log(emission[state, word_map[parsed_sentence[0]]])
        else:
            log_prob[0, state] += np.log(emission[state, num_words])

    for word_idx in range(1, sentence_len):
        for state_curr in range(num_pos):
            for state_prev in range(num_pos):
                new_log_prob = log_prob[word_idx - 1, state_prev] + np.log(
                    transition[state_prev, state_curr]
                )
                if parsed_sentence[word_idx] in word_map:
                    new_log_prob += np.log(
                        emission[state_curr, word_map[parsed_sentence[word_idx]]]
                    )
                else:
                    new_log_prob += np.log(emission[state_curr, num_words])
                ini_log_prob = log_prob[word_idx, state_curr]
                if ini_log_prob == 0 or new_log_prob > ini_log_prob:
                    log_prob[word_idx, state_curr] = new_log_prob
                    prev[word_idx, state_curr] = state_prev
        # print(log_prob)

    # Backtracking for the tags
    rev_path = [np.argmax(log_prob[-1])]
    for word_idx in range(sentence_len - 2, -1, -1):
        rev_path.append(prev[word_idx, rev_path[-1]])

    tags = []

    for tag_val in rev_path[::-1]:
        tags.append(pos_keys[tag_val])

    return tags


for word, _ in tagged_sentences[0]:
    print(word, end=" ")
print()
for _, tag in tagged_sentences[0]:
    print(tag, end=" ")
print()
for i in viterbi(
    datasets[0].transition_matrix,
    datasets[0].emission_matrix,
    "The Fulton County Grand Jury said Friday an investigation of Atlanta's recent primary election produced `` no evidence '' that any irregularities took place .",
):
    print(i, end=" ")
