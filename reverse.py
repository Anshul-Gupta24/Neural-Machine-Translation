# Function to reverse words of string
 
def reverseWords(input):
     
    # split words of string separated by space
    inputWords = input.split(" ")
 
    # reverse list of words
    # suppose we have list of elements list = [1,2,3,4], 
    # list[0]=1, list[1]=2 and index -1 represents
    # the last element list[-1]=4 ( equivalent to list[3]=4 )
    # So, inputWords[-1::-1] here we have three arguments
    # first is -1 that means start from last element
    # second argument is empty that means move to end of list
    # third arguments is difference of steps
    inputWords=inputWords[-1::-1]
 
    # now join words with space
    output = ' '.join(inputWords)
     
    return output

indata = open('train.txt', 'r').read()
inlines = indata.splitlines()
inlines = [w+" <eos>" for w in inlines]
rev = [reverseWords(sentence) for sentence in inlines]
op = open('train_rev2.txt','w')
for sentence in rev:
	op.write(sentence)
	op.write('\n')
