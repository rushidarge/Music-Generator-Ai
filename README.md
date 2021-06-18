# Music-Generator-Ai

Rushikesh Darge
1 Follower
About
Follow
Sign in

Get started


Music Generation Ai
Rushikesh Darge
Rushikesh Darge

Jun 8·5 min read


Photo by Arseny Togulev on Unsplash
‘The art of ordering tones or sound in succession, in combination is music.’ -Definition of music by a dictionary
Traditionally, music was treated as an analog signal and was generated manually. But now we have many others ways to generate music. Now we can create new music without any instrument, we can use notation like midi, ABC notation and create new music like in research paper. And if we use that notation to find patterns and predict new patterns that awesome !!!
Enough Background now let’s jump into the project with great diagrams…
Contents
Music Representation
Music Dataset
Data Processing
Model
Prediction
Summary
Music Representation
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
efe edB | d2d def | gfe edB |1 dBA ABd :|2 dBA AFD |] 
</score>
Image Form of above notationsImage Form of above notations
Image Form of above music notation
For more, you can learn here
Music Dataset
In this project we are going to use “ABC version of the Nottingham Music Database” it contains over 1000 Folk Tunes stored in a special text format.
To build our model that learns from this notation first we have to convert this text format dataset to numerical so the computer can understand and learn from it.
Data Preprocessing
Here we converting text to numbers
{char:num for (char,num) in enumerate(sorted(set(notes)))}
output>>
{0: ‘\n’,
1: ‘ ‘,
2: ‘!’,
3: ‘“‘,
4: ‘#’,
...}

Image representation of Data Preprocessing
char_to_int = dict((num,char) for char,num in enumerate(pitches))
int_to_char = dict((char,num) for char,num in enumerate(pitches))
Preparing data to feed model
Char_to_int consists dictionary that we can map numbers using character, and in int_to_char we can map the character by using the number we use while predicting the output. Because we provide numbers to model give output in number so again to convert in character we use int_to_char.

To convert our one-line dataset to the sequence of data we need to do some processing on it. For that we take the length of 100 characters from 0 to 100 array,101 character is output for that array, then go to next now 1 to 101 array and 102 is output for this array same operation performed on all.
sequence_length = 100
network_input = []
network_output = []
for i in range(0,len(notes)-sequence_length):
seq_in = notes[i:i+sequence_length]
seq_out = notes[i+sequence_length]
network_input.append([char_to_int[i] for i in seq_in])
network_output.append([char_to_int[seq_out]])
Reshaping data
For deep learning, we require 3 dimension dataset so we convert our 2 dimension data to 3 dimension
network_input = np.reshape(network_input,n_pattern,sequence_length,1))
network_input = network_input / np.float(n_vocab)
network_output = to_categorical(network_output)
Model
Our data is the time-series format. In our data, the previous notes decide the next note. So for that, we are going to use the Long short term memory(LSTM ) is an artificial recurrent neural network (RNN) architecture. It is specially made for the time series data format so it performs well on our dataset.

a modified little bit from Colah
Here we can see that the single-cell of LSTM. To understand LSTM deeply you can read a famous blog from Christopher Olah. our final model looks like this

Our LSTM Model
Many to one LSTM

Many-to-One LSTM
There are different-different types of LSTM model like One-to-Many, Many-to-One, and Many-to-Many
But, in our project, we use a many-to-one sequence because, we have a sequence of data as input, which we have in the form of notes, and we predict the next note.
model = Sequential()
model.add(Embedding(93,32,input_length=100))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(n_vocab))
model.add(Activation(‘softmax’))
model.compile(loss=’categorical_crossentropy’, optimizer=’adam’,metrics=[‘accuracy’])
model.summary()
Model: “sequential”
_________________________________________________________________
Layer (type) Output Shape Param #
=================================================================
embedding (Embedding) (None, 100, 32) 2976
_________________________________________________________________
lstm (LSTM) (None, 100, 256) 295936
_________________________________________________________________
dropout (Dropout) (None, 100, 256) 0
_________________________________________________________________
lstm_1 (LSTM) (None, 100, 256) 525312
_________________________________________________________________
dropout_1 (Dropout) (None, 100, 256) 0
_________________________________________________________________
lstm_2 (LSTM) (None, 256) 525312
_________________________________________________________________
dropout_2 (Dropout) (None, 256) 0
_________________________________________________________________
dense (Dense) (None, 93) 23901
_________________________________________________________________
activation (Activation) (None, 93) 0
=================================================================
Total params: 1,373,437
Trainable params: 1,373,437
Non-trainable params: 0
________________________________________________________________
Optimizer
RMSprop stands for Root Mean Square Propagation.it restricts the oscillations in the vertical direction. Therefore, we can increase our learning rate and our algorithm could take larger steps in the horizontal direction converging faster.
Output

Since our model takes input that has a sequence of notations and then predicts the next notation. if we want to create a long notation then we have to run model prediction in a loop. You can easily understand by seeing the diagram.
The first input is just “ABC” then we predict the next notation, then we give (input + new notation) to the model as an input. we run this in a loop. and we get output “ABCDEF”
pattern = network_input[np.random.randint(0,len(network_input))]
prediction_output = []
# generate 200 notes
for i in tqdm(range(200)):
 prediction_input = np.reshape(pattern, (1, len(pattern), 1))
 prediction_input = prediction_input / float(n_vocab)
prediction = model.predict(prediction_input, verbose=0)
index = np.argmax(prediction)
 result = int_to_char[index]
 prediction_output.append(result)
pattern = np.append(pattern,index)
 pattern = pattern[1:len(pattern)]
‘ ‘.join([str(elem) for elem in prediction_output])
Summary
By doing this project I understand how to solve time series problems and also use LSTM. and also how sensitive LSTM models is by just changing few parameters it gives result drastically changes and it also requires a lot of epochs and also time. My Github repo.
That was an awesome experience project. I think we are moving forward to just creating numerical predictions, to the model that creates music that touches people, hearts.
Thanks for reading….
Rushikesh Darge
Follow
50


50



Lstm
Deep Learning
Music Generation
Artificial Intelligence
More from Rushikesh Darge
Follow
More From Medium
Google Open Sourced this Architecture for Massively Scalable Reinforcement Learning Models
Jesus Rodriguez in DataSeries

Computing MFCCs voice recognition features on ARM systems
Antoine CHONÉ in Linagora LABS

Which One Should You choose? GAN or VAE? Part-I
Shuo Li

Why Overfitting is a Bad Idea and How to Avoid It (Part 2: Overfitting in virtual assistants)
Andrew R. Freed in IBM Data and AI

Let’s Read A Story: talking to books using semantic similarity
ml5.js in ml5js

Journey Through the World of NLP -Natural Language Processing! — Part-1
Dinesh Kumar in The Startup

How Word-embeddings evolved to learn social biases and how to improve it to forget them.
Jay Chakalasiya in Analytics Vidhya

Markov model for image context understanding
Yefeng Xia

About

Help

Legal
