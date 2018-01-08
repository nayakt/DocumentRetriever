import os
import sys
import datetime
import math
import operator
import nltk
from bisect import bisect_left
import numpy as np


# Binary search in a sorted list
def bin_search(sorted_list: list, element):
    i = bisect_left(sorted_list, element)
    if i != len(sorted_list) and sorted_list[i] == element:
        return True
    return False


# Load inverted index along with positional index for only those non-stop words appear in questions
# Loading entire inverted index will consume too much memory space
def load_inverted_index(ind_dir, qs_vocab: list):
    inv_ind = dict()
    index_file = 'inverted_index.txt'
    with open(os.path.join(ind_dir, index_file), 'r') as reader:
        line = reader.readline()
        word = ''
        index_lst = list()
        word_in_vocab = False
        while line != '':
            line = line.strip()
            parts = line.split()
            if len(parts) == 1:
                if word_in_vocab:
                    inv_ind[word] += index_lst
                word = parts[0]
                index_lst = list()
                if word in qs_vocab:
                    word_in_vocab = True
                    inv_ind[word] = []
                else:
                    word_in_vocab = False
            elif word_in_vocab:
                doc_id = int(parts[0])
                freq = int(parts[1])
                pos_lst = list()
                for index in range(2, len(parts)):
                    pos_lst.append(int(parts[index]))
                index_lst.append((doc_id, freq, pos_lst))
            line = reader.readline()
    if word_in_vocab:
        inv_ind[word] = index_lst
    return inv_ind


# Load corpus statistics like average document length etc.
def load_corpus_stat(ind_dir):
    file_reader = open(os.path.join(ind_dir, 'corpus_stats.txt'), 'r')
    lines = file_reader.readlines()
    file_reader.close()
    doc_cnt = int(lines[0].strip().split('=')[1])
    avg_dl = int(lines[1].strip().split('=')[1])
    min_l = int(lines[2].strip().split('=')[1])
    max_l = int(lines[3].strip().split('=')[1])
    return doc_cnt, avg_dl, min_l, max_l


# Load pre-defined stop words
def load_stop_words():
    s_words = list()
    file_reader = open('../data/stopwords.txt', 'r')
    lines = file_reader.readlines()
    for line in lines:
        s_words.append(line.replace('\n', '').lower())
    file_reader.close()
    return s_words


# Parse the question
def parse_query(qs, s_words):
    qs_words = list()
    words = qs.split()
    for word in words:
        if word not in s_words:
            qs_words.append(word)
    return qs_words


# Calculate idf of those non-stop words appearing in questions
def calculate_idf(vocab: list, inv_ind: dict, doc_cnt: int):
    idf = dict()
    for word in vocab:
        word_index = list()
        if word in list(inv_ind.keys()):
            word_index = inv_ind[word]
        doc_freq = len(word_index)
        idf[word] = math.log(1+(doc_cnt - doc_freq + 0.5)/(doc_freq + 0.5))
    return idf


# Load statistics for every file like number of token etc.
# Also calculate the length normalization factor for every file
def load_file_stats(ind_dir, avg_dl):
    filename = 'files_stats.txt'
    docs_info = dict()
    len_norm = dict()
    doc_len_ceil = math.ceil(avg_dl * 2.72)         # e = 2.72
    with open(os.path.join(ind_dir, filename), 'r') as reader:
        line = reader.readline()
        while line != "":
            line = line.strip()
            parts = line.split('\t')
            doc_id = int(parts[0])
            docs_info[doc_id] = (parts[1], int(parts[2]), int(parts[3]))
            if docs_info[doc_id][2] <= doc_len_ceil:
                len_norm[doc_id] = 1.0
            else:
                len_norm[doc_id] = math.log(docs_info[doc_id][2]/avg_dl)
            line = reader.readline()
    return docs_info, len_norm


# If topic for a question is provided, then a boolean retrieval will be done based on non-stop words in topic
# Only those documents containing at least one non-stop words from the topic will be ranked based on the question
def get_initial_docs_set(topic_words: list, inv_ind: dict):
    topic_docs = list()
    count_arr = np.zeros(doc_count, dtype=np.int)
    for word in topic_words:
        word_index = list()
        if word in inv_ind.keys():
            word_index = inv_ind[word]
        for index_info in word_index:
            doc_id = int(index_info[0])
            count_arr[doc_id] += 1
        word_stem = get_stem(word)
        if word != word_stem:
            if word_stem in inv_ind.keys():
                word_index = inv_ind[word_stem]
            for index_info in word_index:
                doc_id = int(index_info[0])
                count_arr[doc_id] += 1
    for i in range(0, doc_count):
        if count_arr[i] > 0:
            topic_docs.append(i)
    return topic_docs


# If topic for a question is provided, then a boolean retrieval will be done based on non-stop words in topic
# Only those documents containing at least one non-stop words from the topic will be ranked based on the question
def get_topic_based_docs_set(topic_words: list, inv_ind: dict):
    topic_docs = list()
    count_arr = np.zeros(doc_count, dtype=np.int)
    for word in topic_words:
        match_arr = np.zeros(doc_count, dtype=np.int)
        word_index = list()
        if word in inv_ind.keys():
            word_index = inv_ind[word]
        for index_info in word_index:
            doc_id = int(index_info[0])
            match_arr[doc_id] = 1
            count_arr[doc_id] += 1
        word_stem = get_stem(word)
        if word != word_stem:
            if word_stem in inv_ind.keys():
                word_index = inv_ind[word_stem]
            for index_info in word_index:
                doc_id = int(index_info[0])
                if match_arr[doc_id] == 0:
                    count_arr[doc_id] += 1
    for i in range(0, doc_count):
        if count_arr[i] == len(topic_words):
            topic_docs.append(i)
    return topic_docs


# Return the top ranked documents for a question
def get_top_docs(topic_words: list, qs_words: list, key_words: list, inv_ind: dict, idf: dict,
                 docs_stat: dict, len_norm: dict):
    min_doc_set_len = 100
    bigram_dist = 3
    score = dict()
    exact_word_overlap_count = dict()
    stemmed_word_overlap_count = dict()
    doc_qs_word_pos_map = dict()
    prev_pos_list = dict()
    prev_word_pos = -1
    prev_word_stem = ""
    stem_match_weight = 0.8
    # Get the document set from topic based boolean retrieval
    if len(topic_words) > 0:
        target_words = list()
        for tup in topic_words:
            target_words.append(tup[1])
        doc_set = get_topic_based_docs_set(target_words, inv_ind)
        log_writer.write('All Topic words based boolean retrieved set:' + str(len(doc_set)) + '\n')
        print('All Topic words based boolean retrieved set:', len(doc_set))
        if len(doc_set) < min_doc_set_len:
            doc_set = get_initial_docs_set(target_words, inv_ind)
            log_writer.write('Any Topic words based boolean retrieved set:' + str(len(doc_set)) + '\n')
            print('Any Topic words based boolean retrieved set:', len(doc_set))
    else:
        target_words = list()
        for word_boost_tup in qs_words:
            target_words.append(word_boost_tup[1])
        doc_set = get_initial_docs_set(target_words, inv_ind)
        print('Question words based boolean retrieved set:', len(doc_set))
        log_writer.write('Question words based boolean retrieved set:' + str(len(doc_set)) + '\n')
        if len(doc_set) < min_doc_set_len and len(key_words) > 0:
            for tup in key_words:
                target_words.append(tup[0])
            doc_set = get_initial_docs_set(target_words, inv_ind)
            print('Question words and additional key words based boolean retrieved set:', len(doc_set))
            log_writer.write('Question words and additional key words based boolean retrieved set' +
                             str(len(doc_set)) + '\n')
    # Initialize the score for every documents in the document set
    for doc_id in doc_set:
        score[doc_id] = 0.0
        exact_word_overlap_count[doc_id] = 0
        stemmed_word_overlap_count[doc_id] = 0
        doc_qs_word_pos_map[doc_id] = []
    # For every non-stop word in question update the score for documents
    for word_boost_tup in qs_words:
        word = word_boost_tup[1]
        word_stem = get_stem(word)
        boost = word_boost_tup[2]
        word_pos = word_boost_tup[0]

        stem_index = list()
        if word_stem in list(inv_ind.keys()):
            stem_index = inv_ind[word_stem]
        for index_info in stem_index:
            doc_id = int(index_info[0])
            # If the document is not in the boolean retrieval set, then don't consider it
            if not bin_search(doc_set, doc_id):
                continue
            word_doc_freq = int(index_info[1])
            word_doc_freq_norm = math.log(1 + word_doc_freq)
            # Update the score based on term frequency of the word stem
            score[doc_id] += idf[word_stem] * boost * stem_match_weight * word_doc_freq_norm
            stemmed_word_overlap_count[doc_id] += 1
            # Update the score based on word stem proximity
            if word_pos - prev_word_pos <= bigram_dist and prev_word_stem != word_stem and doc_id in prev_pos_list.keys() and \
                    prev_pos_list[doc_id][0] == prev_word_stem and len(prev_pos_list[doc_id][1]) > 0:
                proximity_score = get_proximity_score(docs_stat[doc_id][0], prev_word_stem,
                                                      prev_pos_list[doc_id][1], word, index_info[2])
                score[doc_id] += proximity_score
            prev_pos_list[doc_id] = (word_stem, index_info[2])
            doc_qs_word_pos_map[doc_id].append((word_stem, index_info[2]))

        word_index = list()
        if word in list(inv_ind.keys()):
            word_index = inv_ind[word]

        for index_info in word_index:
            doc_id = int(index_info[0])
            # If the document is not in the boolean retrieval set, then don't consider it
            if not bin_search(doc_set, doc_id):
                continue
            word_doc_freq = int(index_info[1])
            word_doc_freq_norm = math.log(1 + word_doc_freq)
            # Update the score based on term frequency of the word
            score[doc_id] += (idf[word] * (1 - stem_match_weight) + (idf[word] - idf[word_stem]) *
                              stem_match_weight) * word_doc_freq_norm * boost
            exact_word_overlap_count[doc_id] += 1

        prev_word_pos = word_pos
        prev_word_stem = word_stem

    # Update score based on topic words
    prev_pos_list = dict()
    prev_word_pos = -1
    prev_word_stem = ""
    for word_boost_tup in topic_words:
        word = word_boost_tup[1]
        word_stem = get_stem(word)
        boost = word_boost_tup[2]
        word_pos = word_boost_tup[0]

        stem_index = list()
        if word_stem in list(inv_ind.keys()):
            stem_index = inv_ind[word_stem]
        for index_info in stem_index:
            doc_id = int(index_info[0])
            # If the document is not in the boolean retrieval set, then don't consider it
            if not bin_search(doc_set, doc_id):
                continue
            word_doc_freq = int(index_info[1])
            word_doc_freq_norm = math.log(1 + word_doc_freq)
            # Update the score based on term frequency of the word stem
            score[doc_id] += idf[word_stem] * boost * stem_match_weight * word_doc_freq_norm
            stemmed_word_overlap_count[doc_id] += 1
            # Update the score based on word stem proximity
            if word_pos - prev_word_pos <= bigram_dist and prev_word_stem != word_stem and doc_id in prev_pos_list.keys() and \
                    prev_pos_list[doc_id][0] == prev_word_stem and len(prev_pos_list[doc_id][1]) > 0:
                proximity_score = get_proximity_score(docs_stat[doc_id][0], prev_word_stem,
                                                      prev_pos_list[doc_id][1], word, index_info[2])
                score[doc_id] += proximity_score
            prev_pos_list[doc_id] = (word_stem, index_info[2])
            doc_qs_word_pos_map[doc_id].append((word_stem, index_info[2]))

        word_index = list()
        if word in list(inv_ind.keys()):
            word_index = inv_ind[word]

        for index_info in word_index:
            doc_id = int(index_info[0])
            # If the document is not in the boolean retrieval set, then don't consider it
            if not bin_search(doc_set, doc_id):
                continue
            word_doc_freq = int(index_info[1])
            word_doc_freq_norm = math.log(1 + word_doc_freq)
            # Update the score based on term frequency of the word
            score[doc_id] += (idf[word] * (1 - stem_match_weight) + (idf[word] - idf[word_stem]) *
                              stem_match_weight) * word_doc_freq_norm * boost
            exact_word_overlap_count[doc_id] += 1

        prev_word_pos = word_pos
        prev_word_stem = word_stem

    # Update score based on additional key words
    if len(key_words) > 0:
        for tup in key_words:
            word = tup[0]
            boost = tup[1]
            word_stem = get_stem(word)
            stem_index = list()
            if word_stem in list(inv_ind.keys()):
                stem_index = inv_ind[word_stem]
            for index_info in stem_index:
                doc_id = int(index_info[0])
                if not bin_search(doc_set, doc_id):
                    continue
                word_doc_freq = int(index_info[1])
                word_doc_freq_norm = math.log(1 + word_doc_freq)
                score[doc_id] += idf[word_stem] * boost * word_doc_freq_norm

    # Update the score based on number of overlapping words between question and documents (repetition not considered)
    for doc_id in doc_set:
        # Normalize the score based on length normalization factor
        score[doc_id] /= len_norm[doc_id]
        score[doc_id] += get_context_span_score(doc_qs_word_pos_map[doc_id])
        score[doc_id] += math.log(1 + math.pow(exact_word_overlap_count[doc_id], 2)) * (1 - stem_match_weight)
        score[doc_id] += math.log(1 + math.pow(stemmed_word_overlap_count[doc_id], 2)) * stem_match_weight

    # Sort the documents based on score
    sorted_score = sorted(score.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_score, exact_word_overlap_count, stemmed_word_overlap_count


def get_context_span_score(word_pos_map: list):
    word_cnt = len(word_pos_map)
    unique_word_pos_map = []
    unique_words = list()
    for tup in word_pos_map:
        word = tup[0]
        if word not in unique_words:
            unique_words.append(word)
            unique_word_pos_map.append(tup)
    if len(unique_words) < 2:
        return 0.0
    merge_word_pos_list = list()
    word = unique_word_pos_map[0][0]
    pos_list = unique_word_pos_map[0][1]
    for pos in pos_list:
        merge_word_pos_list.append((word, pos))
    for i in range(1, len(unique_word_pos_map)):
        merge_word_pos_list = get_merge_list(merge_word_pos_list, unique_word_pos_map[i])
    min_span = get_min_span(unique_words, merge_word_pos_list)
    ctx_span_score = math.pow(word_cnt, 2) / max(1.0, math.log(min_span))
    return ctx_span_score


def get_min_span(unique_words, merge_word_pos_list):
    min_span = sys.maxsize
    word_match = dict()
    for i in range(0, len(merge_word_pos_list) - len(unique_words) + 1):
        start_pos = merge_word_pos_list[i][1]
        for word in unique_words:
            word_match[word] = 0
        for j in range(i, len(merge_word_pos_list)):
            cur_word = merge_word_pos_list[j][0]
            cur_pos = merge_word_pos_list[j][1]
            word_match[cur_word] = 1
            total = 0
            for word in unique_words:
                total += word_match[word]
            if total == len(unique_words) and cur_pos - start_pos + 1 < min_span:
                min_span = cur_pos - start_pos + 1
                break
    return min_span


def get_merge_list(merge_word_pos_list, word_pos_list):
    word = word_pos_list[0]
    pos_list = word_pos_list[1]
    merge_list = list()
    ind_1 = 0
    ind_2 = 0
    while ind_1 < len(merge_word_pos_list) and ind_2 < len(pos_list):
        if merge_word_pos_list[ind_1][1] < pos_list[ind_2]:
            merge_list.append(merge_word_pos_list[ind_1])
            ind_1 += 1
        elif merge_word_pos_list[ind_1][1] > pos_list[ind_2]:
            merge_list.append((word, pos_list[ind_2]))
            ind_2 += 1
        else:
            log_writer.write('Error in get_merge_list function\n')
            merge_list.append(merge_word_pos_list[ind_1])
            ind_1 += 1
            merge_list.append((word, pos_list[ind_2]))
            ind_2 += 1

    while ind_1 < len(merge_word_pos_list):
        merge_list.append(merge_word_pos_list[ind_1])
        ind_1 += 1
    while ind_2 < len(pos_list):
        merge_list.append((word, pos_list[ind_2]))
        ind_2 += 1
    return merge_list


# This function calculated the proximity score if two non-stop words of a question appear close in a document
def get_proximity_score(doc, prev_word, prev_pos_list: list, cur_word, cur_pos_list: list):
    distances = list()
    # max_d represents the maximum distance allowed between which
    # if two consecutive non-stop keyword appear, add extra score
    max_d = 10
    pos_ind_1 = 0
    pos_ind_2 = 0
    min_d = sys.maxsize
    # Determine the minimum distance between position of two consecutive not-stop query word in document
    # Store the minimum distance for multiple occurrences
    while pos_ind_1 < len(prev_pos_list) and pos_ind_2 < len(cur_pos_list):
        if cur_pos_list[pos_ind_2] < prev_pos_list[pos_ind_1]:
            pos_ind_2 += 1
            if min_d < sys.maxsize:
                distances.append(min_d)
            min_d = sys.maxsize
        # Following condition is not a possible scenario
        # As two word can't appear at same position of a document
        elif cur_pos_list[pos_ind_2] == prev_pos_list[pos_ind_1]:
            pos_ind_1 += 1
            pos_ind_2 += 1
            distances.append(min_d)
            min_d = sys.maxsize
            print(prev_pos_list)
            print(cur_pos_list)
            log_writer.write(doc + '\n')
            log_writer.write(prev_word + ':' + str(prev_pos_list) + '\n')
            log_writer.write(cur_word + ':' + str(cur_pos_list) + '\n')
            log_writer.write('Not Possible' + '\n')
        else:
            cur_d = cur_pos_list[pos_ind_2] - prev_pos_list[pos_ind_1]
            if cur_d < min_d:
                min_d = cur_d
                pos_ind_1 += 1
    if min_d < sys.maxsize:
        distances.append(min_d)
    sum_d = 0
    for d in distances:
        if d > max_d:
            d = max_d
        # Lesser the distance, higher the score
        # Rather than taking the sum of the minimum distance, add (max_d - d)
        # (max_d - d) will make sure the fact that lesser the distance, higher the score
        sum_d += max_d - d
    return math.log(1 + sum_d)


# Process each question in the input question file
def process_questions(qs_id_list: list, queries: dict, inv_ind: dict, idf: dict, docs_stat: dict,
                      len_norm: dict, top_cnt, out_file):
    writer = open(out_file, 'w')
    for qs_id in qs_id_list:
        print('Processing query: ', qs_id)
        log_writer.write('Processing query: ' + qs_id + '\n')
        topic_words = queries[qs_id][0]
        qs_words = queries[qs_id][1]
        key_words = queries[qs_id][2]
        score, exact_word_overlap_count, stemmed_word_overlap_count = get_top_docs(topic_words, qs_words, key_words,
                                                                                   inv_ind, idf, docs_stat, len_norm)
        cnt = min(len(score), top_cnt)
        for i in range(0, cnt):
            doc_id = score[i][0]
            sim_score = score[i][1]
            writer.write(qs_id + '\t' + str(len(qs_words)) + '\t' + str(docs_stat[doc_id][0]) + '\t' +
                         str(exact_word_overlap_count[doc_id]) + '\t' + str(stemmed_word_overlap_count[doc_id])
                         + '\t' + str(sim_score) + '\n')
        if cnt == 0:
            writer.write(qs_id + '\t' + 'NoDocFound' + '\n')
    writer.close()


# Process a single question for interactive retrieval system
def process_single_question(qs_id, qs_words, topic_words, key_words, inv_ind: dict,
                            idf: dict, docs_stat: dict, len_norm: dict, top_cnt):
    ans_file_name = '../data/temp/rank_' + qs_id + '.txt'
    writer = open(ans_file_name, 'w')
    print('Processing query: ', qs_id)
    score, exact_word_overlap_count, stemmed_word_overlap_count = get_top_docs(topic_words, qs_words, key_words,
                                                                               inv_ind, idf, docs_stat, len_norm)
    cnt = min(len(score), top_cnt)
    for i in range(0, cnt):
        doc_id = score[i][0]
        sim_score = score[i][1]
        writer.write(qs_id + '\t' + str(len(qs_words)) + '\t' + str(docs_stat[doc_id][0]) + '\t' +
                     str(exact_word_overlap_count[doc_id]) + '\t' + str(
            stemmed_word_overlap_count[doc_id]) + '\t' + str(sim_score) + '\n')
    writer.close()


# Load a single query. Load the question, topic, additional key words
def load_single_query(question: str, s_words: list, qs_vocab: list):
    question_parts = question.strip().split('\t')
    qs_id = question_parts[0].strip()
    qs = question_parts[1]
    qs_tokens = clean(qs)
    qs_words = list()
    word_idx = 0
    additional_vocab = list()
    # Load question key words
    for q_word in qs_tokens:
        q_word_parts = q_word.split('^')
        q_word = q_word_parts[0]
        q_word_boost = 1
        if len(q_word_parts) > 1:
            q_word_boost = int(q_word_parts[1])
        if q_word.lower() not in s_words:
            qs_words.append((word_idx, q_word, q_word_boost))
            if q_word not in qs_vocab and q_word not in additional_vocab:
                additional_vocab.append(q_word)
            word_stem = get_stem(q_word)
            if word_stem not in qs_vocab and word_stem not in additional_vocab:
                additional_vocab.append(word_stem)
        word_idx += 1
    # Load topic words
    topic_words = list()
    key_words = list()
    topic = question_parts[2].strip()
    if topic != 'Nothing':
        topic_tokens = clean(topic)
        for t_word in topic_tokens:
            if t_word.lower() not in s_words:
                topic_words.append(t_word)
                key_words.append((t_word, 1))
                if t_word not in qs_vocab and t_word not in additional_vocab:
                    additional_vocab.append(t_word)
                word_stem = get_stem(t_word)
                if word_stem not in qs_vocab and word_stem not in additional_vocab:
                    additional_vocab.append(word_stem)
            word_idx += 1
    # Load additional key words

    add_keys = question_parts[3].strip()
    if add_keys != 'Nothing':
        add_keys_tokens = clean(add_keys)
        for token in add_keys_tokens:
            token_parts = token.split('^')
            token_word = token_parts[0].strip()
            token_boost = 1
            if len(token_parts) > 1:
                token_boost = int(token_parts[1])
            if token_word.lower() not in s_words:
                key_words.append((token_word, token_boost))
                if token_word not in qs_vocab and token_word not in additional_vocab:
                    additional_vocab.append(token_word)
                word_stem = get_stem(token_word)
                if word_stem not in qs_vocab and word_stem not in additional_vocab:
                    additional_vocab.append(word_stem)

    return qs_id, qs_words, topic_words, key_words, additional_vocab


# clean string if necessary
def clean(sent: str):
    tokens = nltk.word_tokenize(sent)
    return tokens


# Get the stem form of a word
def get_stem(word: str):
    try:
        word_stem = stemmer.stem(word)
    except:
        word_stem = word
        log_writer.write('Error generated from Porter Stemmer:' + word + '\n')
        print(word, ' :Error generated from Porter Stemmer')
    return word_stem


# Load all the question from file. Determine the vocabulary to load index and calculate IDF later
def load_queries(qs_file: str, s_words: list):
    qs_id_list = list()
    qs_vocab = dict()
    queries = dict()
    reader = open(qs_file)
    lines = reader.readlines()
    # Read all question in the file
    for line in lines:
        line = line.strip()
        line_parts = line.split('\t')
        qs_id = line_parts[0]
        qs_id_list.append(qs_id)
        qs = line_parts[1]
        qs_tokens = clean(qs)
        qs_words = list()
        word_idx = 0
        # Load question words
        for q_word in qs_tokens:
            q_word_parts = q_word.split('^')
            q_word = q_word_parts[0]
            q_word_boost = 1
            if len(q_word_parts) > 1:
                q_word_boost = int(q_word_parts[1])
            if q_word.lower() not in s_words:
                qs_words.append((word_idx, q_word, q_word_boost))
                if q_word not in list(qs_vocab.keys()):
                    qs_vocab[q_word] = 1
                word_stem = get_stem(q_word)
                if word_stem not in list(qs_vocab.keys()):
                    qs_vocab[word_stem] = 1
            word_idx += 1
        # Load topic words
        word_idx = 0
        topic_words = list()
        topic = line_parts[2].strip()
        if topic != 'Nothing':
            topic_tokens = clean(topic)
            for t_word in topic_tokens:
                if t_word.lower() not in s_words:
                    topic_words.append((word_idx, t_word, 1))
                    if t_word not in list(qs_vocab.keys()):
                        qs_vocab[t_word] = 1
                    word_stem = get_stem(t_word)
                    if word_stem not in list(qs_vocab.keys()):
                        qs_vocab[word_stem] = 1
                word_idx += 1
        # Load additional key words
        additional_key_words = list()
        add_keys = line_parts[3].strip()
        if add_keys != 'Nothing':
            add_keys_tokens = clean(add_keys)
            for token in add_keys_tokens:
                token_parts = token.split('^')
                token_word = token_parts[0].strip()
                token_boost = 1
                if len(token_parts) > 1:
                    token_boost = int(token_parts[1])
                if token_word.lower() not in s_words:
                    additional_key_words.append((token_word, token_boost))
                    if token_word not in list(qs_vocab.keys()):
                        qs_vocab[token_word] = 1
                    word_stem = get_stem(token_word)
                    if word_stem not in list(qs_vocab.keys()):
                        qs_vocab[word_stem] = 1
        queries[qs_id] = (topic_words, qs_words, additional_key_words)
    return qs_id_list, queries, list(qs_vocab.keys())


if __name__ == "__main__":
    index_dir = ""
    qs_file_path = ""
    top = 10
    output_file = ""
    cmd = ' '.join(sys.argv[1:])
    for arg_index in range(1, len(sys.argv), 2):
        if sys.argv[arg_index] == '-top':
            top = int(sys.argv[arg_index + 1])
        elif sys.argv[arg_index] == '-index':
            index_dir = sys.argv[arg_index + 1]
        elif sys.argv[arg_index] == '-qs':
            qs_file_path = sys.argv[arg_index + 1]
        elif sys.argv[arg_index] == '-output':
            output_file = sys.argv[arg_index + 1]

    if index_dir == "" or qs_file_path == "" or output_file == "":
        print('Wrong parameters provided......')
        exit(0)

    log_writer = open('search-log.txt', 'a')
    log_writer.write('search started:' + str(datetime.datetime.now()) + '\n')
    log_writer.write(cmd + '\n')

    stemmer = nltk.PorterStemmer()
    start = datetime.datetime.now()
    stop_words = load_stop_words()
    print('Reading queries......')
    qs_ids, query_list, query_vocab = load_queries(qs_file_path, stop_words)
    print('Reading corpus stats......')
    doc_count, avgDL, minDL, maxDL = load_corpus_stat(index_dir)

    print('Reading inverted index......')
    start = datetime.datetime.now()
    inverted_index = load_inverted_index(index_dir, query_vocab)
    end = datetime.datetime.now()
    print('Index reading time : ', end - start)

    print('Calculating IDF values......')
    start = datetime.datetime.now()
    query_idf = calculate_idf(query_vocab, inverted_index, doc_count)
    end = datetime.datetime.now()
    print('IDF calculation time : ', end - start)

    print('Reading file stats......')
    start = datetime.datetime.now()
    docs_statistics, docs_length_norm_factors = load_file_stats(index_dir, avgDL)
    end = datetime.datetime.now()
    print('File stats reading time : ', end - start)

    print('Processing queries......')
    start = datetime.datetime.now()
    process_questions(qs_ids, query_list, inverted_index, query_idf, docs_statistics,
                      docs_length_norm_factors, top, output_file)
    end = datetime.datetime.now()
    print('Queries processing time : ', end-start)
    log_writer.write('search ended:' + str(datetime.datetime.now()) + '\n\n\n')
    log_writer.close()

    # Interactive retrieval system
    # while 1 == 1:
    #     q = input("Enter question:")
    #     if q.lower() == 'quit':
    #         break
    #     sg_qs_id, sg_qs_words, sg_topic_words, sg_key_words, \
    #         sg_additional_vocab = load_single_query(q, stop_words, query_vocab)
    #     for w in sg_additional_vocab:
    #         w_index = list()
    #         if w in list(inverted_index.keys()):
    #             w_index = inverted_index[w]
    #         d_freq = len(w_index)
    #         query_idf[w] = math.log(1 + (doc_count - d_freq + 0.5) / (d_freq + 0.5))
    #
    #     process_single_question(sg_qs_id, sg_qs_words, sg_topic_words, sg_key_words, inverted_index,
    #                             query_idf, docs_statistics, docs_length_norm_factors, top)
