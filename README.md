## Encoder-Decoder model inspired from Cho et al. (2014) implemented in tensorflow

#### The architecture consists of an Encoder GRU, a Decoder GRU and a Projection Layer.

#### The procedure is as follows:

#### We first make an embedding for both the source and target language to get the vector representation for every word. 

#### The input sequence (which is now tensor of tensors after embedding) is passed to the Encoder which generates the context.

#### This context, along with the target sequence (also a tensor of tensors) is passed onto the Decoder which gives us an output sequence (tensor of tensors).

#### We then run every tensor in the output sequence though a projection layer to get a probability distribution over the vocabulary. The output word is picked by sampling across this distribution.

#### During the test phase, we feed in the generated output rather than the target word back into the Decoder. </br> </br>

### Running the Code:

#### (Source language in 'train.txt', Target language in 'target.txt')
#### ```>> python addtokens.py```
#### ```>> python nmt.py```
####  </br> 

####   

#### Note:

#### addtokens.py adds a start token at the start of every target sentence.
#### For a sanity check, you can use reverse.py to generate the reverse of every source sentence as the target language. </br> </br>

### References
#### Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078.
