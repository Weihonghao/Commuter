

if __name__ == "__main__":
    #question_file = open("question.txt",'r')
    question_train = open("data/train.from",'w')
    question_val = open("data/val.from",'w')
    question_test = open("data/test.from",'w')
    
    
    #answer_file = open("answer.txt",'r')
    answer_train = open("data/train.to",'w')
    answer_val = open("data/val.to",'w')
    answer_test = open("data/test.to",'w')
    
    
    num_train = 5000
    num_val = 1200 + num_train

    count = 0
    with open("question.txt") as file:
    	for line in file:
    		if count < num_train:
    			question_train.write(line)
    		elif count < num_val:
    			question_val.write(line)
    		else:
    			question_test.write(line)
            count +=1

    print count
    question_train.close()
    question_val.close()
    question_test.close()


    count = 0
    with open("answer.txt") as file:
    	for line in file:
    		if count < num_train:
    			answer_train.write(line)
    		elif count < num_val:
    			answer_val.write(line)
    		else:
    			answer_test.write(line)
            count += 1

    print count
    answer_train.close()
    answer_val.close()
    answer_test.close()






