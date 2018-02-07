'''
	add start and end tokens to file
'''

indata = open('train.txt', 'r').read()
inlines = indata.splitlines()
inlines = ["<st> "+w+" <eos>" for w in inlines]
op = open('train2.txt', 'w')
for sentence in inlines:
        op.write(sentence)

