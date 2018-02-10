'''
	add start and end tokens to file
'''

indata = open('train.txt', 'r').read()
inlines = indata.splitlines()
inlines = ["<eos> "+w for w in inlines]
op = open('train_rev.txt', 'w')
for sentence in inlines:
        op.write(sentence)
	op.write('\n')
