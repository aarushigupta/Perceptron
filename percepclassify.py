import numpy as np
import sys
from perceplearn import data_cleanup, split_into_words, readFile, remove_stop_words

def readModelParameters():
    unique_words = {}
    bias_TF = 0
    bias_PN = 0
    filename = sys.argv[1]

    with open(filename,'r') as f:
        fileContents = f.readlines()

    SEPARATOR = "******######******######******######******######******"
    flag = 0

    for index in range(len(fileContents)):
        line = fileContents[index].strip()
        if line == SEPARATOR:
            flag += 1
            continue

        if flag == 1:
            unique_words_list = line.split("\t")
            unique_words[unique_words_list[0]] = int(unique_words_list[1])
            # words_labels_mat = [[] for x in range(len(uniqueWords))]


        if flag == 2:
            weights_true_fake = map(float,line.split("\t"))

        if flag == 3:
            weights_pos_neg = map(float,line.split("\t"))
        
        if flag == 4:
            bias_TF = float(line.strip())
        
        if flag == 5:
            bias_PN = float(line.strip())

    weights_true_fake = np.array(weights_true_fake)
    weights_pos_neg = np.array(weights_pos_neg)

    
    return unique_words, weights_true_fake, weights_pos_neg, bias_TF, bias_PN



def label_test_data(review, unique_words, weights_pos_neg, weights_true_fake, bias_TF, bias_PN):
    
    one_hot_vector = np.zeros(len(unique_words))
    for word in review[1:]:
        if word in unique_words:
            one_hot_vector[unique_words[word]] += 1
                
    res_true_fake = np.sum(weights_true_fake * one_hot_vector) + bias_TF
    
    res_pos_neg = np.sum(weights_pos_neg * one_hot_vector) + bias_PN
    
    if res_true_fake < 0:
        test_label_true_fake = 'Fake'
    else:
        test_label_true_fake = 'True'
        
    if res_pos_neg < 0:
        test_label_pos_neg = 'Neg'
    else:
        test_label_pos_neg = 'Pos'
        
    return test_label_true_fake, test_label_pos_neg
            

if __name__ == '__main__':


    unique_words, weights_true_fake, weights_pos_neg, bias_TF, bias_PN = readModelParameters()
    test_data = readFile(sys.argv[2])
    test_doc = [[] for x in range(len(test_data))]
    words_in_reviews_test = ['' for x in range(len(test_data))]
    for review_index in range(len(test_data)):
        test_data[review_index] = data_cleanup(test_data[review_index])
        words_in_reviews_test[review_index] = split_into_words(test_data[review_index])
        words_in_reviews_test[review_index] = remove_stop_words(words_in_reviews_test[review_index])
        test_doc[review_index].append(words_in_reviews_test[review_index][0])
        test_label_true_fake, test_label_pos_neg = label_test_data(words_in_reviews_test[review_index], unique_words, weights_pos_neg, weights_true_fake, bias_TF,bias_PN)
        test_doc[review_index].append(test_label_true_fake)
        test_doc[review_index].append(test_label_pos_neg)
    

    output = ""
    for row in test_doc:
        output += " ".join(map(str,row)) + "\n"

    all_words = unique_words.keys()
    all_words.sort()
    all_words = ' '.join(all_words)
    with open ('words', 'w') as f:
        f.write(all_words)
        f.close()
        
    filename = "percepoutput.txt"
    with open (filename, 'w') as f:
        f.write(output)
        f.close()

