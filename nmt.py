'''
	Machine Translation using an Encoder Decoder Architecture (Cho et al, 2014)
'''

from __future__ import print_function
import tensorflow as tf
import numpy as np
import re
import zero_pad as zp


#process data

indata = open('train2.txt', 'r').read()
outdata = open('train_rev.txt', 'r').read()
inlines = indata.split(' <eos>')
inlines = inlines[:-1]
outlines = outdata.split(' <st>')
outlines = outlines[:-1]
words_in = indata.split(' <eos>')
words_in = indata.split(' ')
words_out = outdata.split(' <st>')
words_out = outdata.split(' ')
inp=[s.split(' ') for s in inlines]
out=[s.split(' ') for s in outlines]

vocabulary_in = set(words_in)
vocabulary_out = set(words_out)
inp_sz=len(vocabulary_in)+1	#for null character
out_sz=len(vocabulary_out)+1
print("inp_sz is")
print(inp_sz)

word_ind_in = {w:i for i,w in enumerate(vocabulary_in)}
ind_word_in = {i:w for i,w in enumerate(vocabulary_in)}
sv = ind_word_in[0]
ind_word_in[0]='null'
ind_word_in[len(ind_word_in)]=sv
word_ind_in[sv]=len(word_ind_in)
word_ind_in['null']=0
word_ind_out = {w:i for i,w in enumerate(vocabulary_out)}
ind_word_out = {i:w for i,w in enumerate(vocabulary_out)}
sv = ind_word_out[0]
ind_word_out[0]='null'
ind_word_out[len(ind_word_out)]=sv
word_ind_out[sv]=len(word_ind_out)
word_ind_out['null']=0

int_inp = [0]*len(inlines)
int_out = [0]*len(outlines)

for x,i in zip(xrange(len(inlines)),inp):
	int_inp[x] = [word_ind_in[w] for w in i]

for x,i in zip(xrange(len(outlines)),out):
	int_out[x] = [word_ind_out[w] for w in i]


sequence_lengths_inp = [len(seq) for seq in inp]
sequence_lengths_out = [len(seq) for seq in out]

# zero padding

final_inp, max_inp_len = zp.zero_pad(int_inp)
final_out, max_out_len = zp.zero_pad(int_out)



# hyperparameters

num_epochs = 10000
batch_size = 1
#truncated_backprop_length = 20
num_batches = len(inlines)//batch_size
state_size = 200
embedding_size = 200



#cell = tf.nn.rnn_cell.BasicLSTMCell(state_size, forget_bias=1.0, state_is_tuple=False)
with tf.variable_scope('encoder'):
	cell = tf.nn.rnn_cell.GRUCell(state_size)
with tf.variable_scope('decoder'):
	cell2 = tf.nn.rnn_cell.GRUCell(state_size)

init_state = tf.placeholder(tf.float32, [batch_size, state_size])
init_state2 = tf.placeholder(tf.float32, [batch_size, state_size])
W2 = tf.Variable(np.random.rand(state_size, out_sz),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1, out_sz)), dtype=tf.float32)
	
batchX_placeholder = tf.placeholder(tf.int32, [batch_size, max_inp_len])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, max_out_len])


word_embeddings_inp = tf.get_variable("word_embeddings_inp", [inp_sz, embedding_size])
word_embeddings_out = tf.get_variable("word_embeddings_out", [out_sz, embedding_size])

batchX = tf.nn.embedding_lookup(word_embeddings_inp, batchX_placeholder)
batchY = tf.nn.embedding_lookup(word_embeddings_out, batchY_placeholder[:,:-1])


#inputs_series = tf.split(batchX, truncated_backprop_length, 1)
#inputs_series = [tf.squeeze(i,1) for i in inputs_series]
#inputs_series = tf.unstack(batchX, axis=1)
#print(tf.shape(inputs_series))

labels_series = tf.unstack(batchY_placeholder[:,1:], axis=1)
#inputs_decoder = tf.unstack(batchY, axis=1)

max_out_len=max_out_len-1


seq_length_inp = tf.placeholder(tf.int32, [None])
seq_length_out = tf.placeholder(tf.int32, [None])



#encoder
_, current_state = tf.nn.dynamic_rnn(cell, batchX, sequence_length = seq_length_inp, initial_state = init_state, scope='encoder')


#decoder
#concatenate encoder last hidden state with every input to decoder

context = [current_state for r in xrange(max_out_len)]
context = tf.stack(context,axis=1)
batchY = tf.concat([batchY, context],2)

states_series, current_state2 = tf.nn.dynamic_rnn(cell2, batchY, sequence_length = seq_length_out, initial_state = init_state2, scope='decoder')


states_series = tf.unstack(states_series, axis=1)

logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]


# calculate losses
	
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=label) for logit, label in zip(logits_series,labels_series)]
total_loss = tf.reduce_mean(losses)

#train_step = tf.train.AdagradOptimizer(1e-1).minimize(total_loss)
train_step = tf.train.AdamOptimizer(learning_rate=0.0001, epsilon = 0.0000008).minimize(total_loss)


# sample char according to probabilities of output vector

def get_word(inp):
	
	op_ind = np.zeros(len(inp))
	c = 0
	for i in inp:
		
		op_ind[c] = int(np.random.choice(range(out_sz), p=i))
		c+=1

	return op_ind
	
# get next word

def sample(curr, curr2, ip):

	#cont = [current_state]
	#cont = tf.stack(cont,axis=1)
	ipt = tf.concat([ip, curr],1)
	print(tf.shape(ipt))
	ipt, cr2 = cell2(ipt, curr2)
	ip = tf.matmul(ipt,W2) + b2
			
	return tf.nn.softmax(ip), cr2


def softmax(inp):
	return (np.exp(inp) / np.sum(np.exp(inp)))



curr_state = tf.placeholder(tf.float32, [batch_size, state_size])
curr_state2 = tf.placeholder(tf.float32, [batch_size, state_size])
ipy = tf.placeholder(tf.int32, [batch_size])	
ipy2 = tf.nn.embedding_lookup(word_embeddings_out, ipy)
op, fin_state = sample(curr_state, curr_state2, ipy2)


# run session

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    loss_list = []
    print("numbatches : ",num_batches)
    #print("total inp: ",len(inp))
    #print("total : ",data_len)

    len_data = len(int_inp) - len(int_inp)%batch_size

    x = final_inp[:len_data]
    y = final_out[:len_data]

    for epoch_idx in range(num_epochs):
	
		
	#x = x.reshape((batch_size, -1))  
	#y = y.reshape((batch_size, -1))


	print("New data, epoch", epoch_idx)
	
	for batch_idx in range(num_batches):


	    start_idx = ((batch_idx)*batch_size) 
	    end_idx = ((batch_idx+1)*batch_size) 
	
    	    _current_state = np.zeros((batch_size, state_size))
     	    _current_state2 = np.zeros((batch_size, state_size))
	    batchX = x[start_idx:end_idx]
	    batchY = y[start_idx:end_idx]

	    seq_len_inp = sequence_lengths_inp[start_idx:end_idx]
	    seq_len_out = sequence_lengths_out[start_idx:end_idx]
	    seq_len_out = [(s-1) for s in seq_len_out]


	    _total_loss, _train_step, _current_state, _current_state2, _predictions_series = sess.run(
		[total_loss, train_step, current_state, current_state2, predictions_series],
		feed_dict={
		    batchX_placeholder:batchX,
		    batchY_placeholder:batchY,
		    init_state:_current_state,
		    init_state2:_current_state2,
		    seq_length_inp: seq_len_inp,
		    seq_length_out: seq_len_out
		})

	    loss_list.append(_total_loss)

	    if batch_idx%100 == 0:
		print("Step",batch_idx, "Loss", _total_loss)
		print("")
		print("")



    	# get translation of sample sentence every epoch

    	# get random first char; testing on first sentence... can change

	_ip = np.zeros((batch_size, max_inp_len), dtype=int)
	c=0
	seq_len_inp = [sequence_lengths_inp[0]] * batch_size
	for i in xrange(batch_size):
			_ip[c] = np.copy(x[0])
			c+=1

	#print(_ip.shape)
	test_current_state = np.zeros((batch_size, state_size))
	test_current_state2 = np.zeros((batch_size, state_size))
   	#_current_state2 = np.copy(_current_state)
	print(test_current_state.shape)
	test_current_state = sess.run(
			current_state,
			feed_dict={
		    		batchX_placeholder:_ip,
		    		init_state:test_current_state,
				seq_length_inp: seq_len_inp
			})

	string=""
	_ipy = [0]*batch_size
	for i in xrange(len(_ipy)):
			_ipy[i] = word_ind_in['<eos>']

   	for i in xrange(max_out_len):

			#print(_current_state2)
			#print(len(test_current_state))
			#print("_ip")
			#print(_ip[0])
		
			p, test_current_state2 = sess.run([op, fin_state], feed_dict={curr_state:test_current_state, curr_state2:test_current_state2, ipy: _ipy})
		
			_ipy = get_word(p)
			c = ind_word_out[_ipy[0]]
			string+=c
			string+=" "

	print(string)
 	print("")
	print("")
