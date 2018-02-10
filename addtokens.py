'''
	add start and end tokens to file
'''

indata = open('target.txt', 'r').read()
inlines = indata.splitlines()
inlines = ["<eos> "+w for w in inlines]
op = open('target2.txt', 'w')
for sentence in inlines:
        op.write(sentence)
	op.write('\n')
