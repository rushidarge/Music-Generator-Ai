
<h1> Music Generation Ai</h1>
<h5>[Medium article link](https://rushikeshdarge.medium.com/music-generation-ai-fc0241c59372)</h5>

![Photo by Arseny Togulev on Unsplash](https://miro.medium.com/max/2800/0*H7fvQVu0tpAsPYe1)

> **‘The art of ordering tones or sound in succession, in combination is music.’ -Definition of music by a dictionary**

Traditionally, music was treated as an analog signal and was generated manually. But now we have many others ways to generate music. Now we can create new music without any instrument, we can use notation like midi, ABC notation and create new music like in [research paper](https://iarjset.com/papers/lstm-based-music-generation-system/). And if we use that notation to find patterns and predict new patterns that awesome !!!

Enough Background now let’s jump into the project with great diagrams…

Contents
========

*   Music Representation
*   Music Dataset
*   Data Processing
*   Model
*   Prediction
*   Summary

Music Representation
====================

Our data set is in ABC notation which is one of the music notations for computers. In basic form it uses the letter notation with a–g, A–G, and z, to represent the corresponding notes and rests, with other elements used to place added value on these — sharp, flat, raised or lowered octave, the note length, key, and ornamentation. You can see the example below

<score lang="ABC">  
X:1  
T:The Legacy Jig  
M:6/8  
L:1/8  
R:jig  
K:G  
GFG BAB | gfg gab | GFG BAB | d2A AFD |  
GFG BAB | gfg gab | age edB |1 dBA AFD :|2 dBA ABd |:  
efe edB | dBA ABd | efe edB | gdB ABd |  
efe edB | d2d def | gfe edB |1 dBA ABd :|2 dBA AFD |\]   
</score>

![Image Form of above notations](https://miro.medium.com/max/1400/0*IXo9X45MEXdxrcwh)

Image Form of above music notation

For more, you can learn [here](https://abcnotation.com/learn)

Music Dataset
=============

In this project we are going to use “[ABC version of the Nottingham Music Database](http://abc.sourceforge.net/NMD/)” it contains over 1000 Folk Tunes stored in a special text format.

To build our model that learns from this notation first we have to convert this text format dataset to numerical so the computer can understand and learn from it.

Data Preprocessing
==================

Here we converting text to numbers

**{char:num for (char,num) in enumerate(sorted(set(notes)))}  
**output>>  
{0: ‘\\n’,  
1: ‘ ‘,  
2: ‘!’,  
3: ‘“‘,  
4: ‘#’,  
...}

![](https://miro.medium.com/max/2560/0*SgEzrf000zuPJVou)

Image representation of **Data Preprocessing**

char\_to\_int = dict((num,char) for char,num in enumerate(pitches))int\_to\_char = dict((char,num) for char,num in enumerate(pitches))

Preparing data to feed model
----------------------------

Char\_to\_int consists dictionary that we can map numbers using character, and in int\_to\_char we can map the character by using the number we use while predicting the output. Because we provide numbers to model give output in number so again to convert in character we use int\_to\_char.

![](https://miro.medium.com/max/2560/0*b1dy3i_sOtRR0bGy)

To convert our one-line dataset to the sequence of data we need to do some processing on it. For that we take the length of 100 characters from 0 to 100 array,101 character is output for that array, then go to next now 1 to 101 array and 102 is output for this array same operation performed on all.

sequence\_length = 100  
network\_input = \[\]  
network\_output = \[\]  
for i in range(0,len(notes)-sequence\_length):  
seq\_in = notes\[i:i+sequence\_length\]  
seq\_out = notes\[i+sequence\_length\]  
network\_input.append(\[char\_to\_int\[i\] for i in seq\_in\])  
network\_output.append(\[char\_to\_int\[seq\_out\]\])

Reshaping data
--------------

For deep learning, we require 3 dimension dataset so we convert our 2 dimension data to 3 dimension

network\_input = np.reshape(network\_input,n\_pattern,sequence\_length,1))network\_input = network\_input / np.float(n\_vocab)  
network\_output = to\_categorical(network\_output)

Model
=====

Our data is the time-series format. In our data, the previous notes decide the next note. So for that, we are going to use the [Long short term memory(LSTM )](https://en.wikipedia.org/wiki/Long_short-term_memory) is an artificial recurrent neural network (RNN) architecture. It is specially made for the time series data format so it performs well on our dataset.

![](https://miro.medium.com/max/1600/1*5goFqf2xxgBD7UJk4WXfgw.png)

a modified little bit from [Colah](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

Here we can see that the single-cell of LSTM. To understand LSTM deeply you can read a famous blog from [Christopher Olah](https://colah.github.io/about.html). our final model looks like this

![](https://miro.medium.com/max/2560/1*ATDOYGBXrQiQHA8oXasVEQ.png)

Our LSTM Model

Many to one LSTM
----------------

![](https://miro.medium.com/max/2560/0*BUOJyrBcZ22TrHG5)

Many-to-One LSTM

There are different-different types of LSTM model like One-to-Many, Many-to-One, and Many-to-Many

But, in our project, we use a many-to-one sequence because, we have a sequence of data as input, which we have in the form of notes, and we predict the next note.

model = Sequential()model.add(Embedding(93,32,input\_length=100))model.add(LSTM(256, return\_sequences=True))model.add(Dropout(0.2))model.add(LSTM(256, return\_sequences=True))model.add(Dropout(0.2))model.add(LSTM(256))model.add(Dropout(0.2))model.add(Dense(n\_vocab))model.add(Activation(‘softmax’))model.compile(loss=’categorical\_crossentropy’, optimizer=’adam’,metrics=\[‘accuracy’\])model.summary()Model: “sequential”  
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_  
Layer (type) Output Shape Param #  
\=================================================================  
embedding (Embedding) (None, 100, 32) 2976  
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_  
lstm (LSTM) (None, 100, 256) 295936  
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_  
dropout (Dropout) (None, 100, 256) 0  
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_  
lstm\_1 (LSTM) (None, 100, 256) 525312  
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_  
dropout\_1 (Dropout) (None, 100, 256) 0  
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_  
lstm\_2 (LSTM) (None, 256) 525312  
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_  
dropout\_2 (Dropout) (None, 256) 0  
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_  
dense (Dense) (None, 93) 23901  
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_  
activation (Activation) (None, 93) 0  
\=================================================================  
Total params: 1,373,437  
Trainable params: 1,373,437  
Non-trainable params: 0  
\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

Optimizer
---------

RMSprop stands for Root Mean Square Propagation.it restricts the oscillations in the vertical direction. Therefore, we can increase our learning rate and our algorithm could take larger steps in the horizontal direction converging faster.

Output
======

![](https://miro.medium.com/max/2560/0*1rjxdFdpfelb1s3C)

Since our model takes input that has a sequence of notations and then predicts the next notation. if we want to create a long notation then we have to run model prediction in a loop. You can easily understand by seeing the diagram.

_The first input is just “ABC” then we predict the next notation, then we give (input + new notation) to the model as an input. we run this in a loop. and we get output “ABCDEF”_

pattern = network\_input\[np.random.randint(0,len(network\_input))\]  
prediction\_output = \[\]\# generate 200 notes  
for i in tqdm(range(200)):  
 prediction\_input = np.reshape(pattern, (1, len(pattern), 1))  
 prediction\_input = prediction\_input / float(n\_vocab)prediction = model.predict(prediction\_input, verbose=0)index = np.argmax(prediction)  
 result = int\_to\_char\[index\]  
 prediction\_output.append(result)pattern = np.append(pattern,index)  
 pattern = pattern\[1:len(pattern)\]  
‘ ‘.join(\[str(elem) for elem in prediction\_output\])

Summary
=======

By doing this project I understand how to solve time series problems and also use LSTM. and also how sensitive LSTM models is by just changing few parameters it gives result drastically changes and it also requires a lot of epochs and also time. My [Github repo](https://github.com/rushidarge/Music-Generator-Ai).

That was an awesome experience project. I think we are moving forward to just creating numerical predictions, to the model that creates music that touches people, hearts.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Thanks for reading….
