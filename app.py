from flask import Flask,render_template,request
import cv2
from keras.applications.resnet import ResNet50
import numpy as np
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import LSTM,GRU
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model
from tqdm import tqdm

vocab = np.load('vocab.npy',allow_pickle=True)
vocab = vocab.item()
inv_vocab = {v:k for k,v in vocab.items()}
resnet = ResNet50(include_top = False,  weights = 'imagenet',input_shape = (224,224,3), pooling = 'avg')


embedding_size = 128
max_len = 40
vocab_size = len(vocab) + 1
image_model = Sequential()
image_model.add(Dense(embedding_size,input_shape = (2048,),activation = 'relu'))
image_model.add(RepeatVector(max_len))

language_model = Sequential()
language_model.add(Embedding(input_dim = vocab_size,output_dim = embedding_size,input_length = max_len))
language_model.add(LSTM(256,return_sequences = True))
language_model.add(TimeDistributed(Dense(embedding_size)))

conca = Concatenate()([image_model.output,language_model.output])

x = LSTM(128,return_sequences = True)(conca)
x = LSTM(512,return_sequences = False)(x)
x = Dense(vocab_size)(x)

out = Activation('softmax')(x)

model = Model(inputs = [image_model.input,language_model.input],outputs = out)
model.compile(loss = 'categorical_crossentropy',optimizer = 'RMSProp',metrics = ['accuracy'])


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

model.load_weights('mine_model_weights.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/after',methods = ['GET','POST'])
def after():
    global model,vocab,inv_vocab

    
    file = request.files['file1']
    file.save('static/file.jpg')
    
    img = cv2.imread('static/file.jpg')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img,(224,224))
    img = img.reshape(1,224,224,3)
    
    features = resnet.predict(img).reshape(1,2048)
    
    text_inp = ['startofseq']

    count = 0
    caption = ''
    while count < 25:
        count += 1

        encoded = []
        for i in text_inp:
            encoded.append(vocab[i])

        encoded = [encoded]

        encoded = pad_sequences(encoded, padding='post', truncating='post', maxlen=max_len)


        prediction = np.argmax(model.predict([features, encoded]))

        sampled_word = inv_vocab[prediction]

        caption = caption + ' ' + sampled_word
            
        if sampled_word == 'endofseq':
            break

        text_inp.append(sampled_word)
    
    return render_template('after.html',final = caption)    

if __name__ == '__main__':
    app.run()