import librosa
import soundfile,time
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib as plt

## file_name=os.path.basename(file)
## num_label = file_name.split("-")[3]
## print("num_label",num_label)
   
    
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result

emotions={
  '1':'Rooster',
  '2':'Pig',
  '3':'Cow',
  '4':'Frog',
  '5':'Cat',
  '6':'Hen',
  '7':'Insects',
  '8':'Sheep',
  '9':'Crow',
  '10':'Rain'
}

# Emotions to observe
observed_emotions=['Rooster', 'Pig', 'Cow', 'Frog','Cat','Hen','Insects','Sheep','Crow','Rain']

def load_data(test_size=0.25):
    x,y=[],[]
    for file in glob.glob("train*\\*"):
        file_name=os.path.basename(file)
        file_name=os.path.splitext(file_name)[0]
        num_label = file_name.split("-")[3]
##        filename_wo_ext = filename.with_suffix('')

##        print("num_label",num_label)
        emotion=emotions[num_label]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(num_label)
    return train_test_split(np.array(x), np.array(y), test_size=test_size)


x_train,x_test,y_train,y_test=load_data(test_size=0.25)
##input data##############                                        
file1="test/1-40730-A-1.wav"
feature1=extract_feature(file1, mfcc=True, chroma=True, mel=True)
print("feature shape",feature1.shape)

##print("**********************MLR*************************")
print("y_train",y_train)
import numpy as np
import sklearn.metrics
print("**************RNN*******************")
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear')
classifier.fit(x_train, y_train)

Y_Pred = classifier.predict(x_test)
print("prediction",Y_Pred)

# Making the Confusion Matrix 

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, Y_Pred)
print (cm)
#acuracy for our model
from sklearn.metrics import accuracy_score
print ("accuracy",accuracy_score(y_test, Y_Pred))

from tensorflow import keras
from keras.layers import Dense, Flatten, LSTM
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.models import Sequential

##
##print("Training labels shape: y_train.shape:{}".format(y_train.shape))

x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1)
x_test = x_test.reshape(x_test.shape[0], x_train.shape[1],1)

##Convert class vectors to binary class matrices. This uses 1 hot encoding.
y_train_binary = keras.utils.to_categorical(y_train)
y_test_binary = keras.utils.to_categorical(y_test)
print(y_test_binary)
##
##
##
input_shape = (180,1)
batch_size = 20
num_classes = 11
epochs = 50



##print("Y TestBinary:",y_test_binary)

model_rnn = Sequential()
model_rnn.add(LSTM(units = 2,
                   input_shape=input_shape,
                   activation = 'relu',
                   return_sequences=True))
model_rnn.add(Flatten())
model_rnn.add(Dense(64, activation='relu'))
model_rnn.add(Dense(num_classes, activation='softmax'))

model_rnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model_rnn.summary()

history=model_rnn.fit(x_train, y_train_binary,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test_binary))

y_pred_rnn=model_rnn.predict_classes(x_test)
print(y_pred_rnn)
feature1 = feature1.reshape(1,180,1)








pred_rnn=model_rnn.predict_classes(feature1)
print("new input ",pred_rnn)
