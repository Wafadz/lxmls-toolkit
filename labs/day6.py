# Load Part-of-Speech data
from lxmls.readers.pos_corpus import PostagCorpusData
wsj_data = PostagCorpusData()

# RNN configuration
from lxmls.deep_learning.numpy_models.rnn import NumpyRNN

embedding_size = 50   # Size of word embeddings
hidden_size = 20      # size of hidden layer

np_rnn = NumpyRNN(
    input_size=wsj_data.input_size,
    embedding_size=embedding_size,
    hidden_size=hidden_size,
    output_size=wsj_data.output_size,
    learning_rate=0.5
)

num_epochs = 300
model = np_rnn

# Get batch iterators for train and test
train_batches = wsj_data.batches('train', batch_size=1)
dev_batches = wsj_data.batches('dev', batch_size=1)

# Epoch loop
for epoch in range(num_epochs):

    # Batch loop
    for batch in train_batches:
        model.update(input=batch['input'], output=batch['output'])

    # Prediction for this epoch
    hits = 0
    num_words = 0
    for num, batch in enumerate(dev_batches):
        hat_y = model.predict(input=batch['input'])
        hits += sum(hat_y == batch['output'])
        num_words += hat_y.shape[0]

    # Evaluation
    accuracy = 100*hits*1./num_words

    # Inform user
    print("Epoch %d: dev accuracy %2.2f %%" % (epoch+1, accuracy))
