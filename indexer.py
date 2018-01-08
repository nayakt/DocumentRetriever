import os
import sys
import datetime
import nltk
import math
import codecs


# Following function returns all the files in a given directory which will be indexed
# Assumption is that all the files in that directory are text file
def get_files(dir_name):
    files = list()
    dirs = list()
    dirs.append(os.path.abspath(dir_name))
    while len(dirs) > 0:
        cur_dir = dirs.pop(0)
        cur_dir_entries = os.listdir(cur_dir)
        for entry in cur_dir_entries:
            full_entry = os.path.join(cur_dir, entry)
            if os.path.isfile(full_entry):
                files.append(full_entry)
            else:
                dirs.append(full_entry)
    files.sort()
    return files


# This function load pre-defined stop words
def load_stop_words():
    s_words = list()
    file_reader = open('../data/stopwords.txt')
    lines = file_reader.readlines()
    for line in lines:
        s_words.append(line.replace('\n', '').lower())
    file_reader.close()
    return s_words


# Initialize the index writer based on index mode
# For mode = 0, index writer will starts from scratch
# For mode = 1, index writer will update the old index
def init_index(ind_dir, ind_mode):
    if ind_mode == 0 or (ind_mode == 1 and not os.path.exists(ind_dir)):
        if not os.path.exists(ind_dir):
            os.mkdir(ind_dir)
        corp_stat_writer = open(os.path.join(ind_dir, 'corpus_stats.txt'), 'w')
        file_stat_writer = open(os.path.join(ind_dir, 'files_stats.txt'), 'a')
        inv_ind_writer = open(os.path.join(ind_dir, 'inverted_index.txt'), 'a')
        doc_count = 0
        total_token_count = 0
        min_len = sys.maxsize
        max_len = -1
    else:
        corp_stat_reader = open(os.path.join(ind_dir, 'corpus_stats.txt'), 'r')
        lines = corp_stat_reader.readlines()
        doc_count = int(lines[0].replace('\n', '').split('=')[1])
        total_token_count = int(doc_count * float(lines[1].replace('\n','').split('=')[1]))
        min_len = int(lines[2].replace('\n', '').split('=')[1])
        max_len = int(lines[3].replace('\n', '').split('=')[1])
        corp_stat_reader.close()
        corp_stat_writer = open(os.path.join(ind_dir, 'corpus_stats.txt'), 'w')
        file_stat_writer = open(os.path.join(ind_dir, 'files_stats.txt'), 'a')
        inv_ind_writer = open(os.path.join(ind_dir, 'inverted_index.txt'), 'a')
    return corp_stat_writer, file_stat_writer, inv_ind_writer, min_len, \
        max_len, total_token_count, doc_count


# This function returns the stem form given the surface form of a word
def get_stem(word: str):
    try:
        word_stem = stemmer.stem(word)
    except:
        word_stem = word
        log_writer.write('Error generated from Porter Stemmer: ' + word + '\n')
    return word_stem


# This function creates the index for every file
def create_index(s_words, files, ind_dir, ind_mode, min_size, max_size):
    corp_stat_writer, file_stat_writer, inv_ind_writer, min_len, max_len, \
                total_non_stop_token_count, doc_count = init_index(ind_dir, ind_mode)
    file_id = doc_count
    file_count = 0
    file_tmp_cnt = 0
    corpus_inv_ind = dict()
    file_stats = list()
    # indexing every file in directory
    for f in files:
        try:
            if file_count % 1000 == 0:
                print('Added ', file_count, ' files')
            print('Adding file ', os.path.abspath(f))
        except:
            print('invalid file name')
            continue
        file_reader = codecs.open(f, 'r', 'utf-8', errors='ignore')
        content = file_reader.read().strip()
        file_reader.close()
        # content = str(content.encode(encoding='UTF-8', errors='ignore'))
        # content = content.strip()
        # Tokenize the file
        tokens = nltk.word_tokenize(content)
        # check on file length. Files with unwanted length will not be indexed

        non_stop_token_count = 0
        file_inv_ind = dict()
        # create inverted index for every token in the file along with their position in the file
        for token_count in range(0, len(tokens)):
            token = tokens[token_count]
            if not token.lower() in s_words:
                non_stop_token_count += 1
                token_stem = get_stem(token)
                if token in file_inv_ind.keys():
                    file_inv_ind[token].append(token_count)
                else:
                    file_inv_ind[token] = [token_count]
                if token != token_stem:
                    if token_stem in file_inv_ind.keys():
                        file_inv_ind[token_stem].append(token_count)
                    else:
                        file_inv_ind[token_stem] = [token_count]
        if non_stop_token_count < min_size or non_stop_token_count > max_size:
            continue
        # print('Adding file ', os.path.abspath(f))
        total_non_stop_token_count += non_stop_token_count
        if non_stop_token_count < min_len:
            min_len = non_stop_token_count
        if non_stop_token_count > max_len:
            max_len = non_stop_token_count
        file_stats.append(str(file_id) + "\t" + os.path.abspath(f) + "\t" + str(len(tokens)) +
                               "\t" + str(non_stop_token_count) + "\n")
        # file_stat_writer.write(str(file_id) + "\t" + os.path.abspath(f) + "\t" + str(len(tokens)) +
        #                        "\t" + str(non_stop_token_count) + "\n")
        # Add the inverted index for this file to the global inverted index
        for token in file_inv_ind.keys():
            token_freq = len(file_inv_ind[token])
            inv_ind_line = str(file_id) + " " + str(token_freq) + " "
            for pos in file_inv_ind[token]:
                inv_ind_line += str(pos) + " "
            inv_ind_line = inv_ind_line.strip()
            if token in corpus_inv_ind.keys():
                corpus_inv_ind[token].append(inv_ind_line)
            else:
                corpus_inv_ind[token] = [inv_ind_line]

        file_id += 1
        file_count += 1
        file_tmp_cnt += 1
        if file_tmp_cnt == 100000:
            file_tmp_cnt = 0
            # Write the inverted index to file
            for token in corpus_inv_ind.keys():
                inv_ind_writer.write(token + "\n")
                for inv_ind_line in corpus_inv_ind[token]:
                    inv_ind_writer.write(inv_ind_line + "\n")
            corpus_inv_ind = dict()

            for file_stat in file_stats:
                file_stat_writer.write(file_stat)
            file_stats = list()

    # Write the inverted index to file
    for token in corpus_inv_ind.keys():
        inv_ind_writer.write(token + "\n")
        for inv_ind_line in corpus_inv_ind[token]:
            inv_ind_writer.write(inv_ind_line + "\n")

    for file_stat in file_stats:
        file_stat_writer.write(file_stat)

    file_stat_writer.close()
    doc_count += file_count
    corp_stat_writer.write("doc_count=" + str(doc_count) + "\n")
    avg_dl = int(math.ceil(total_non_stop_token_count * 1.0 / doc_count))
    corp_stat_writer.write("AvgDL=" + str(avg_dl) + "\n")
    corp_stat_writer.write("MinDL=" + str(min_len) + "\n")
    corp_stat_writer.write("MaxDL=" + str(max_len) + "\n")
    corp_stat_writer.close()
    inv_ind_writer.close()


if __name__ == "__main__":
    docs_dir = ""
    index_dir = ""
    max_file_size = sys.maxsize
    min_file_size = 1
    # 0 for new index, 1 for update
    index_mode = 0
    cmd = ' '.join(sys.argv[1:])
    for i in range(1, len(sys.argv), 2):
        if sys.argv[i] == '-docs':
            docs_dir = sys.argv[i+1]
        elif sys.argv[i] == '-index':
            index_dir = sys.argv[i+1]
        elif sys.argv[i] == '-mode':
            index_mode = int(sys.argv[i+1])
        elif sys.argv[i] == '-max_size':
            max_file_size = int(sys.argv[i+1])
        elif sys.argv[i] == '-min_size':
            min_file_size = int(sys.argv[i+1])

    if docs_dir == "" or index_dir == "" or index_mode < 0 or index_mode > 1 or max_file_size < 1 or min_file_size < 1 or min_file_size >= max_file_size:
        print('Wrong parameters provided......')
        exit(0)
    log_writer = open('index-log.txt', 'a')
    log_writer.write('index started:' + str(datetime.datetime.now()) + '\n')
    log_writer.write(cmd + '\n')
    stemmer = nltk.PorterStemmer()
    start = datetime.datetime.now()
    stop_words = load_stop_words()
    files_to_index = get_files(docs_dir)
    create_index(stop_words, files_to_index, index_dir, index_mode, min_file_size, max_file_size)
    end = datetime.datetime.now()
    print('Total time : ', end-start)
    log_writer.write('index ended:' + str(datetime.datetime.now()) + '\n\n\n')
    log_writer.close()
