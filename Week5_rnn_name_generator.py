# -*- coding: utf-8 -*-
# Generating Fantasy Character Names
# This file has been adapted from https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string

#import pdb

#-------------------------------------------------------------------
# Functions to read and clean data

# define the set of available characters
all_letters = string.ascii_letters + " .,;'-"
# each character is represented as an index number  including the SOS (start of seqeuence)
# and EOS (end of sequence) index numbers
n_letters = len(all_letters) + 1 # Plus SOS/EOS marker

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


# Read the list of character sequences
# The fantasy names are taken from https://www.reddit.com/r/DnDBehindTheScreen/comments/50pcg1/a_post_about_names_names_for_speakers_of_the/
filename ="fantasy_names2.txt"
lines = readLines(filename)
all_lines = lines




#-------------------------------------------------------------------
# The current network is a feedforward network having an input for each
# possible character and an output for each possible character and one
# hidden layer.
# If we use this to generate names, we feed in a character, sample from
# the output, then feed the output back to the input.
# This is not ideal, since the only information that the network will
# have about the sequence is the current character.
# We need to instead change the network so that the hidden layer information
# is passed on to the next stage.

import torch
import torch.nn as nn

import torch.nn.functional as F


# Hints: 1. How many layers we need to add to the networks? 
# 2. What is the size of these layers ? 

class myRNN(nn.Module):

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size):

        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            nonlinearity='relu',
            batch_first=True
        )

        self.i2o = nn.Linear(
            input_size + hidden_size,
            output_size
        )

        self.o2o = nn.Linear(
            hidden_size + output_size,
            output_size
        )

        self.dropout = nn.Dropout(0.1)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):

        output_seq, hn = self.rnn(x)

        h_t = output_seq[:, -1, :]
        x_t = x[:, -1, :]

        combined = torch.cat((x_t, h_t), dim=1)

        o_t = F.relu(self.i2o(combined))

        output_combined = torch.cat(
            (h_t, o_t),
            dim=1
        )

        y = self.o2o(output_combined)

        y = self.dropout(y)

        y = self.softmax(y)

        return y, hn

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Net, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input_combined = torch.cat((input, hidden), 1)  
        hidden = F.relu(self.i2h(input_combined))
        output = F.relu(self.i2o(input_combined))
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


#-------------------------------------------------------------------
# Functions used when training the network

import random

# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Get a random line from the list of sequences
def randomTrainingPair():
    #category = randomChoice(all_categories)
    line = randomChoice(all_lines)
    return line

# One-hot matrix of first to last letters (not including EOS) for input
# EOS should not be provided as input.
def inputTensor(line):
    tensor = torch.zeros(len(line) + 1, 1, n_letters) # first position id is for SOS symbol
    li = 0
    tensor[li][0][n_letters-1] = 1 # SOS is first symbol of all seqeunces
    for li in range(len(line)):
        letter = line[li]
        tensor[li+1][0][all_letters.find(letter)] = 1
    return tensor

def inputTensor2(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor


# LongTensor of second letter to end (EOS) for target (all outputs ids)
def targetTensor2(line):
    # find ids for all characters in the sequence
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    # add EOS number to end of character number sequence
    letter_indexes.append(n_letters - 1)
    return torch.LongTensor(letter_indexes)

# LongTensor of first letter to end (EOS) for target (all outputs ids)
# first letter is the targer for SOS
def targetTensor(line):
    # find ids for all characters in the sequence
    letter_indexes = [all_letters.find(line[li]) for li in range(0, len(line))]
    # add EOS number to end of character number sequence
    letter_indexes.append(n_letters - 1)
    return torch.LongTensor(letter_indexes)


#-------------------------------------------------------------------
# Training function
#
# For each training iteration, this train function is called.  Each
# letter is passed to the network one at a time with the next letter
# given as output.
# The loss is computed and parameter updated using gradient descent.

# Use the Negative Log Likelihood Loss
# https://pytorch.org/docs/stable/nn.html#nllloss
criterion = nn.NLLLoss()
# Set learning rate
learning_rate = 0.0005

def train(input_line_tensor, target_line_tensor):

    # structure target to one row for each input
    target_line_tensor.unsqueeze_(-1)
    hidden = rnn.initHidden()

    rnn.zero_grad()

    loss = 0

    # forward pass for each letter in the phrase
    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l

    # compute backpropagation loss
    loss.backward()

    # update weights using gradient descent
    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item() / input_line_tensor.size(0)



#-------------------------------------------------------------------
# Function to keep track of training time

import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


#-------------------------------------------------------------------
# Sample from the network
#
# Sample one letter at a time, given the previous letter to the network.
# The sequence stops when the EOS symbol is found or the max length is reached.

# set the maximum length of a sampled sequences
max_length = 20

# Sample a sequence of letters from the network, starting at the SOS symbol and
# ending at the EOS symbol.
def sample():
    with torch.no_grad():  # no need to track history in sampling
        #category_tensor = categoryTensor(category)
        input = inputTensor('A') # given any letter, all we want inthe SOS symbol
        hidden = rnn.initHidden()

        output_name = ""

        #pdb.set_trace()
        symbol_pos = 0 # set the wanted letter to the first symbol (SOS).
        
        for i in range(max_length):
            # keep samping characters until EOS is sampled (to show end of sequence).
            output, hidden = rnn(input[symbol_pos], hidden)

            # take the most likely letter
            #topv, topi = output.topk(1)
            #topi = topi[0][0]
            # or sample from output distribution
            topi = torch.multinomial(output.exp(), 1) 

            if topi == n_letters - 1:
                # if sampled letter is the EOS marker, finish
                break
            else:
                letter = all_letters[topi]
                output_name += letter
            input = inputTensor(letter)
            symbol_pos = 1 # set the wanted letter to the second symbol (not SOS).
            
        return output_name


#-------------------------------------------------------------------
# Train network
#
# Make a forward pass through the network (by providing a letter), the
# back propagate and update network weights using gradient descent.
# Run for a set number of iterations.

#rnn = Net(n_letters, 128, n_letters)
rnn = RNN(n_letters, 512, n_letters)

#n_iters = 100000
n_iters = 100000
print_every = 1000
plot_every = 500
all_losses = []
total_loss = 0 # Reset every plot_every iters

start = time.time()

for iter in range(1, n_iters + 1):
    line = randomTrainingPair()

    # generate input one hot vectors
    input_line_tensor = inputTensor(line)
    # generate output vector
    target_line_tensor = targetTensor(line)
    
    output, loss = train(input_line_tensor, target_line_tensor)

    total_loss += loss


    if iter % print_every == 0:
        rnn.eval() # turn off dropout
        
        print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

        names_wanted = 3 # print out 3 names
        for a in range(names_wanted):
            x = sample()
            print('Sample output: %s' % x)

        rnn.train() # turn back to training mode (activate dropout)
        
    # keep loss numbers to examine loss decay
    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0


#------------------------------------------------------------------
# After the model is trainied, genreate some names!
rnn.eval() # turn off dropout        

names_wanted = 10 # print out 10 names
for a in range(names_wanted):
    x = sample()
    print('Sample output: %s' % x)
