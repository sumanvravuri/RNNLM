
import scipy.io as sio
import subprocess
import numpy as np
import sys

def read_vocab_to_dict(vocab_list_fn, unknown_tag_included = True):
    """
    given a vocab list with out of vocabulary key and no start and stop tags, returns a vocabulary dictionary to map keys to values
    return vocab_dict
    """
    print "Creating dictionary...", 
    vocab_dict = dict()
    val = 0
    with open(vocab_list_fn) as fpi:
        for line in fpi:
            key = line.strip()
            vocab_dict[key] = val
            val += 1

    vocab_dict['<s>'] = val
    if not unknown_tag_included:
        vocab_dict['<unk>'] = val + 1
    print "DONE"
    return vocab_dict

def max_sequence_len(data_fn):
    max_seq_len = 0

    with open(data_fn) as fpi:
        for line in fpi:
            tokens = line.strip().split(" ")
            if len(tokens) + 2 > max_seq_len: # + 2 for start and stop tags
                max_seq_len = len(tokens) + 2

    return max_seq_len

def read_data_to_mat(data_fn, vocab_dict, unknown_tag_included = True):
    print "reading data to matrix...",
    proc = subprocess.Popen("wc -w " + data_fn + "| awk '{print $1}'", stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    num_words = int(out.strip())
    proc = subprocess.Popen("wc -l " + data_fn + " | awk '{print $1}'", stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    num_sents = int(out.strip()) 
#    num_words += 2 * num_sents #for start and stop tags
    vocab_size = len(vocab_dict)

    max_seq_len = max_sequence_len(data_fn)
    data_mat = np.zeros((max_seq_len, num_sents))
    feature_sequence_lengths = np.zeros((num_sents,), dtype=np.int32)
#    line_num = 0
    sent_num = 0
    frame_in_sent = 0
    with open(data_fn) as fpi:
        for line in fpi:
            tokens = ['<s>'] + line.strip().split(" ")[:-1]
            for token in tokens:
                if not unknown_tag_included:
                    if token not in vocab_dict:
                        token_id = vocab_dict['<unk>']
                    else:
                        token_id = vocab_dict[token]
                else:
                    token_id = vocab_dict[token]
                data_mat[frame_in_sent, sent_num] = token_id
                feature_sequence_lengths[sent_num] += 1 #sent_num, frame_in_sent
#                line_num += 1
                frame_in_sent += 1
            sent_num += 1
            frame_in_sent = 0
    print "DONE"
    return feature_sequence_lengths, data_mat

def read_next_label_to_mat(data_fn, vocab_dict, unknown_tag_included = True):
    print "reading labels to matrix...",
    proc = subprocess.Popen("wc -w " + data_fn + " | awk '{print $1}'", stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    num_words = int(out.strip())
    proc = subprocess.Popen("wc -l " + data_fn + " | awk '{print $1}'", stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    num_sents = int(out.strip()) 
#    num_words += 2 * num_sents #for start and stop tags
    vocab_size = len(vocab_dict)

    label_mat = np.zeros((num_words, 2), dtype=np.int32)
#    frames = np.zeros((num_words, 2), dtype=np.int32)
    line_num = 0
    sent_num = 0
    frame_in_sent = 0
    with open(data_fn) as fpi:
        for line in fpi:
            tokens = line.strip().split(" ")
            for token in tokens:
                if not unknown_tag_included:
                    if token not in vocab_dict:
                        token_id = vocab_dict['<unk>']
                    else:
                        token_id = vocab_dict[token]
                else:
                    token_id = vocab_dict[token]
#                print sent_num, line_num
                label_mat[line_num, :] = sent_num, token_id
                line_num += 1
#                frames[line_num, :] = sent_num, frame_in_sent
#                frame_in_sent += 1
            sent_num += 1
            frame_in_sent = 0
#            token_id = vocab_dict[
#    print label_mat
    print "DONE"
    return label_mat

def save_feature_file(feature_fn, data, fsl):
    sio.savemat(feature_fn, {'features' : data, 'feature_sequence_lengths' : fsl})

def save_label_file(label_fn, labels):
    sio.savemat(label_fn, {'labels' : labels})
                
if __name__ == "__main__":

    if len(sys.argv) != 5 and len(sys.argv) != 6:
        print "USAGE: python %s <input sentences> <vocab dictionary> <output feature file> <output label file> [unk_token_included]" % sys.argv[0]
        sys.exit()

    input_file, vocab_dict_fn, output_feature_fn, output_label_fn = sys.argv[1:5]
    unk_token_included = True
    if len(sys.argv) == 6:
        unk_token_included = sys.argv[5] in ['True', '1', 'T', 't']
    
    print "unknown token is included in training data is", unk_token_included
    vocab_dict = read_vocab_to_dict(vocab_dict_fn, unk_token_included)
    fsl, data = read_data_to_mat(input_file, vocab_dict, unk_token_included)
    save_feature_file(output_feature_fn, data, fsl)
    labels = read_next_label_to_mat(input_file, vocab_dict, unk_token_included)
    save_label_file(output_label_fn, labels)
    
