import numpy as np
import sys

def readFile(file_ext):
	data = None
	with open(file_ext) as f:
		data = f.readlines()  
	return data


def data_cleanup(review):
	symbols_to_remove = [',','.','!','"','@','#','%','^','&','*','(',')','{','}','[',']','|','\\',';',':','<','>','?','/',\
						'`','~','_','+','=',"$"]
	symbols_to_remove += ['0','1','2','3','4','5','6','7','8','9']
	try:
		space_index = review.index(" ")
		review_id = review[:space_index + 1]
	except:
		print "Error in finding space_index in data_cleanup"
	remaining_review = review[space_index :].lower()
	for char in symbols_to_remove:
		remaining_review = remaining_review.replace(char, " ")
	review = review_id + remaining_review

	return review

def split_into_words(data):
	try:
		words_in_reviews = data.strip().split()
	except:
		print "Exception in split_into_words"
	return words_in_reviews


def word_labels(words_in_reviews, unique_words, total_words):


	labels = ['true', 'fake', 'pos', 'neg']
	
	try:
		if words_in_reviews[1] == 'true':
			label_true_fake = 1
		else:
			label_true_fake = -1
	except:
		print "Exception in review[1] in word_labels"

	try:
		if words_in_reviews[2] == 'pos':
			label_pos_neg = 1
		else:
			label_pos_neg = -1
			
	except:
		print "Exception in review[1] in word_labels"

		
	for word in words_in_reviews[3:]:
		if word not in unique_words:
#                 if word.endswith('ing'):
#                     word = word[:-3]
#                 if word.endswith('ed'):
#                      word = word[:-2]
					
			unique_words[word] = total_words
			total_words += 1
				
	
	return unique_words, label_true_fake, label_pos_neg, total_words

def make_one_hot_vector(words_in_reviews, unique_words):
	one_hot_vector = [[0 for y in range(len(unique_words))] for x in range(len(words_in_reviews))]
	for review_index, review in enumerate(words_in_reviews):
		for word in review:
			if word in unique_words:
				one_hot_vector[review_index][unique_words[word]] = 1
	one_hot_vector = np.array(one_hot_vector)
	return one_hot_vector
				


def train_vanilla_perceptron(maxIter,label_true_fake, label_pos_neg, unique_words, one_hot_vector):
	bias_TF = 0
	bias_PN = 0
	weights_true_fake = np.zeros((1,len(unique_words)))
	weights_pos_neg = np.zeros((1,len(unique_words)))
	a_true_fake_inter = np.array((weights_true_fake.shape))
	a_pos_neg_inter = np.array((weights_true_fake.shape))
	
				
	for hyperparam in range(maxIter):
		for index,review in enumerate(one_hot_vector):
			a_true_fake = np.sum(review * weights_true_fake) + bias_TF
			a_pos_neg = np.sum(review * weights_pos_neg) + bias_PN
			
			if label_true_fake[index] * a_true_fake <= 0.0:
				weights_true_fake += label_true_fake[index] * review
				bias_TF += label_true_fake[index]

			if label_pos_neg[index] * a_pos_neg <= 0.0:
				weights_pos_neg += label_pos_neg[index] * review
				bias_PN += label_pos_neg[index]




#     for hyperparam in range(maxIter):
#         for index, review in enumerate(words_in_reviews):
#             one_hot_vector = [0] * len(unique_words)
#             for word in review:
#                 if word in unique_words:
# #                     word_index = unique_words.index(word)
#                     one_hot_vector[unique_words[word]] = 1
					
#             one_hot_vector = np.array(one_hot_vector)
#             a_true_fake = np.sum(one_hot_vector * weights_true_fake) + bias_TF
#             a_pos_neg = np.sum(one_hot_vector * weights_pos_neg) + bias_PN

#             if label_true_fake[index] * a_true_fake <= 0.0:
#                 weights_true_fake += label_true_fake[index] * one_hot_vector
#                 bias_TF += label_true_fake[index]
#             if label_pos_neg[index] * a_pos_neg <= 0.0:
#                 weights_pos_neg += label_pos_neg[index] * one_hot_vector
#                 bias_PN += label_pos_neg[index]

				
	return weights_true_fake, weights_pos_neg, bias_TF, bias_PN


def train_averaged_perceptron(maxIter, label_true_fake, label_pos_neg, unique_words, one_hot_vector):
	bias_TF = 0
	bias_PN = 0
	weights_true_fake = np.zeros((1,len(unique_words)))
	weights_pos_neg = np.zeros((1,len(unique_words)))
	u_true_fake = np.zeros((1,len(unique_words)))
	u_pos_neg = np.zeros((1,len(unique_words)))
	c = 1
	bias_TF_avg = 0
	bias_PN_avg = 0

	
	for hyperparam in range(maxIter):
		for index,review in enumerate(one_hot_vector):
			a_true_fake = np.sum(review * weights_true_fake) + bias_TF
			a_pos_neg = np.sum(review * weights_pos_neg) + bias_PN
			
			if label_true_fake[index] * a_true_fake <= 0.0:
				weights_true_fake += label_true_fake[index] * review
				bias_TF += label_true_fake[index]
				u_true_fake += label_true_fake[index] * c * review
				bias_TF_avg += label_true_fake[index] * c

			if label_pos_neg[index] * a_pos_neg <= 0.0:
				weights_pos_neg += label_pos_neg[index] * review
				bias_PN += label_pos_neg[index]
				u_pos_neg += label_pos_neg[index] * c * review
				bias_PN_avg += label_pos_neg[index] * c

			c += 1

	weights_true_fake_avg = weights_true_fake - (u_true_fake * 1.0 / c)
	weights_pos_neg_avg = weights_pos_neg - (u_pos_neg *1.0 / c)

	return weights_true_fake_avg, weights_pos_neg_avg, bias_TF, bias_PN





	# for hyperparam in range(maxIter):
	#     for index, review in enumerate(words_in_reviews):
	#         one_hot_vector = [0 for x in range(len(unique_words))]
	#         for word in review:
	#             if word in unique_words:
	#                 one_hot_vector[unique_words[word]] = 1
			
	#         one_hot_vector = np.array(one_hot_vector)
	#         a_true_fake = np.sum(one_hot_vector * weights_true_fake) + bias_TF
	#         a_pos_neg = np.sum(one_hot_vector * weights_pos_neg) + bias_PN

	#         if label_true_fake[index] * a_true_fake <= 0.0:
	#             weights_true_fake += label_true_fake[index] * one_hot_vector
	#             bias_TF += label_true_fake[index]
	#         u_true_fake += label_true_fake[index] * c * one_hot_vector
			
	#         if label_pos_neg[index] * a_pos_neg <= 0.0:
	#             weights_pos_neg += label_pos_neg[index] * one_hot_vector
	#             bias_PN += label_pos_neg[index]
	#         u_pos_neg += label_pos_neg[index] * c * one_hot_vector
			
	#         c += 1
			
	weights_true_fake_avg = weights_true_fake - (u_true_fake * 1.0 / c)
	weights_pos_neg_avg = weights_pos_neg - (u_pos_neg *1.0 / c)

	return weights_true_fake_avg, weights_pos_neg_avg, bias_TF, bias_PN
			
		
			
def write_model_parametersVanilla(unique_words, weights_pos_neg, weights_true_fake, bias_TF, bias_PN, filename = 'vanillamodel.txt'):
	SEPARATOR = "******######******######******######******######******\n"
	output = "******######******######******######******######******\n"

	allWords = ""
	for word in unique_words:
		allWords += str(word) + "\t" + str(unique_words[word]) + "\n"

	output += allWords
	output += SEPARATOR

	weight_vals_true_fake = ""
	for weight in weights_true_fake:
		weight_vals_true_fake += '\t'.join(map(str, weight)) + '\n'

	output += weight_vals_true_fake
	output += SEPARATOR

	weight_vals_pos_neg = ""
	for weight in weights_pos_neg:
		weight_vals_pos_neg += '\t'.join(map(str, weight)) + '\n'

	output += weight_vals_pos_neg
	output += SEPARATOR
	
	output += str(bias_TF) + "\n"
	output += SEPARATOR
	
	output += str(bias_PN) + "\n"
	output += SEPARATOR


	with open (filename,'w') as f:
		f.write(output)
		f.close()

def write_model_parametersAveraged(unique_words, weights_pos_neg, weights_true_fake, bias_TF, bias_PN, filename = 'averagedmodel.txt'):
	SEPARATOR = "******######******######******######******######******\n"
	output = "******######******######******######******######******\n"

	allWords = ""
	for word in unique_words:
		allWords += str(word) + "\t" + str(unique_words[word]) + "\n"

	output += allWords
	output += SEPARATOR

	weight_vals_true_fake = ""
	for weight in weights_true_fake:
		weight_vals_true_fake += '\t'.join(map(str, weight)) + '\n'

	output += weight_vals_true_fake
	output += SEPARATOR

	weight_vals_pos_neg = ""
	for weight in weights_pos_neg:
		weight_vals_pos_neg += '\t'.join(map(str, weight)) + '\n'

	output += weight_vals_pos_neg
	output += SEPARATOR
	
	output += str(bias_TF) + "\n"
	output += SEPARATOR
	
	output += str(bias_PN) + "\n"
	output += SEPARATOR


	with open (filename,'w') as f:
		f.write(output)
		f.close()


if __name__ == '__main__':
	data = readFile(sys.argv[1])
	label_true_fake = [0 for x in range(len(data))]
	label_pos_neg = [0 for x in range(len(data))]
	unique_words = {}
	words_in_reviews = [[] for x in range(len(data))]
	total_words = 0

	for review_index in range(len(data)):
		
		data[review_index] = data_cleanup(data[review_index])
		words_in_reviews[review_index] = split_into_words(data[review_index])
		unique_words, label_true_fake[review_index], label_pos_neg[review_index], total_words = word_labels(words_in_reviews[review_index], unique_words, total_words)

	one_hot_vector = make_one_hot_vector(words_in_reviews, unique_words)
	weights_true_fake, weights_pos_neg, bias_TF, bias_PN = train_vanilla_perceptron(40, label_true_fake, label_pos_neg, unique_words, one_hot_vector)
	weights_true_fake_avg, weights_pos_neg_avg, bias_TF_avg, bias_PN_avg = train_averaged_perceptron(40, label_true_fake, label_pos_neg, unique_words, one_hot_vector)


	write_model_parametersVanilla(unique_words, weights_pos_neg, weights_true_fake, bias_TF, bias_PN)
	write_model_parametersAveraged(unique_words, weights_pos_neg_avg, weights_true_fake_avg, bias_TF_avg, bias_PN_avg)
