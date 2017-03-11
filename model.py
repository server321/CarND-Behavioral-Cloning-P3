# Import packages

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
from keras.models import load_model, save_model
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout,Convolution2D,MaxPooling2D,Flatten,Lambda,Cropping2D,ELU
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
import json


# Input data
data_path = 'C:/11SDC'
data_log = '/driving_log.csv'

# Output data
model_json = 'model.json'
model_weights = 'model.h5'

# Read log data cvs file with pandas
column_names = ['center', 'left','right','steering','throttle','brake','speed']
log_data = pd.read_csv(data_path + data_log, names = column_names)

# Load log data to X_train and Y_train
# Split data into train and validation datasets

log_data[['left','center','right']]
X_train = log_data[['left','center','right']]
Y_train = log_data['steering']

X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.2, random_state=15)

# get rid of the pandas index after shuffling

X_center = X_train['center'].as_matrix()
X_left  = X_train['left'].as_matrix()
X_right = X_train['right'].as_matrix()

X_validation   = X_validation['center'].as_matrix()


Y_train = Y_train.as_matrix()


Y_validation   = Y_validation.as_matrix()



Y_train = Y_train.astype(np.float32)
Y_validation   = Y_validation.astype(np.float32)




def read_image(id ,left_center_right,X_center,X_left,X_right,Y_train):
    # assume the side cameras are about 1.2 meters off the center and the offset to the left or right 
    # should be be corrected over the next dist meters, calculate the change in steering control
    # using tan(alpha)=alpha

    offset=1.0 
    dist=20.0
    steering = Y_train[id]
    if left_center_right == 0:
        image = plt.imread(X_left[id].strip(' '))
        dsteering = offset/dist * 360/( 2*np.pi) / 25.0
        steering += dsteering
    elif left_center_right == 1:
        image = plt.imread(X_center[id].strip(' '))
    elif left_center_right == 2:
        image = plt.imread(X_right[id].strip(' '))
        dsteering = -offset/dist * 360/( 2*np.pi)  / 25.0
        steering += dsteering
    else:
        print ('Left_center_right value error:',left_center_right )
    
    return image,steering

def random_crop(image,steering=0.0,tx_lower=-20,tx_upper=20,ty_lower=-2,ty_upper=2,rand=True):
    # we will randomly crop subsections of the image and use them as our data set.
    # also the input to the network will need to be cropped, but of course not randomly and centered.
    shape = image.shape
    col_start,col_end =abs(tx_lower),shape[1]-tx_upper
    horizon=60;
    bonnet=136
    if rand:
        tx= np.random.randint(tx_lower,tx_upper+1)
        ty= np.random.randint(ty_lower,ty_upper+1)
    else:
        tx,ty=0,0
    
    #    print('tx = ',tx,'ty = ',ty)
    random_crop = image[horizon+ty:bonnet+ty,col_start+tx:col_end+tx,:]
    #image = cv2.resize(random_crop,(320,160),cv2.INTER_AREA)
    image = cv2.resize(random_crop,(64,64),cv2.INTER_AREA)
    
    # the steering variable needs to be updated to counteract the shift 
    if tx_lower != tx_upper:
        dsteering = -tx/(tx_upper-tx_lower)/3.0
    else:
        dsteering = 0
    steering += dsteering
    
    return image,steering

def random_shear(image,steering,shear_range):
    rows,cols,ch = image.shape
    dx = np.random.randint(-shear_range,shear_range+1)
    #    print('dx',dx)
    random_point = [cols/2+dx,rows/2]
    pts1 = np.float32([[0,rows],[cols,rows],[cols/2,rows/2]])
    pts2 = np.float32([[0,rows],[cols,rows],random_point])
    dsteering = dx/(rows/2) * 360/(2*np.pi*25.0) / 6.0    
    M = cv2.getAffineTransform(pts1,pts2)
    image = cv2.warpAffine(image,M,(cols,rows),borderMode=1)
    steering +=dsteering
    
    return image,steering

def random_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = 0.8 + 0.4*(2*np.random.uniform()-1.0)    
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def random_flip(image,steering):
    coin=np.random.randint(0,2)
    if coin==0:
        image,steering=cv2.flip(image,1),-steering
    return image,steering
        

def generate_training_example(X_center,X_left,X_right,Y_train):
    id = np.random.randint(0,len(Y_train))
    
    # Randomly choose left, center or right image (0,1,2 correspondently)
    left_center_right = np.random.randint(0,3)


    image,steering = read_image(id, left_center_right,X_center,X_left,X_right,Y_train)

    image,steering = random_shear(image,steering,shear_range=100)
    
    image,steering = random_crop(image,steering,tx_lower=-20,tx_upper=20,ty_lower=-10,ty_upper=10)

    
    image,steering = random_flip(image,steering)

    
    
    image = random_brightness(image)

    
    return image,steering

def get_validation_set(X_validation,Y_validation):
    #X = np.zeros((len(X_validation),160,320,3))
    X = np.zeros((len(X_validation),64,64,3))
    Y = np.zeros(len(X_validation))
    for i in range(len(X_validation)):
        x,y = read_image(i,1,X_validation,X_validation,X_validation,Y_validation)
        X[i],Y[i] = random_crop(x,y,tx_lower=0,tx_upper=0,ty_lower=0,ty_upper=0)
    return X,Y
    

def generate_train_batch(X_center,X_left,X_right,Y_train,batch_size = 32):
    
    #batch_images = np.zeros((batch_size, 160, 320, 3))
    batch_images = np.zeros((batch_size, 64, 64, 3))
    batch_steering = np.zeros(batch_size)
    while 1:
        for i_batch in range(batch_size):
            x,y = generate_training_example(X_center,X_left,X_right,Y_train)
            batch_images[i_batch] = x
            batch_steering[i_batch] = y
        yield batch_images, batch_steering



batch_size=200
train_generator = generate_train_batch(X_center,X_left,X_right,Y_train,batch_size)
X_validation,Y_validation = get_validation_set(X_validation,Y_validation)



def CNN():
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(64,64,3)))
    model.add(Convolution2D(32, 8,8 ,border_mode='same', subsample=(4,4)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 8,8 ,border_mode='same',subsample=(4,4)))
    model.add(Activation('relu',name='relu2'))
    model.add(Convolution2D(128, 4,4,border_mode='same',subsample=(2,2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 2,2,border_mode='same',subsample=(1,1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128))
    model.add(Dense(1))
    model.summary()
    
    return model

def Nvidia():
    model = Sequential()

    # Normalize image
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    #inp = Input(shape=(None, None, 3))
    
    

    # Cropping Layer
    # 50 rows pixels from the top of the image
    # 20 rows pixels from the bottom of the image
    # 0 columns of pixels from the left of the image
    # 0 columns of pixels from the right of the image

    #model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    #crop to Nvidia size 66 x 200
    model.add(Cropping2D(cropping=((70,24), (60,60)), input_shape=(160,320,3)))
    
    #model.add(Lambda(lambda x: ktf.image.resize_images(x, (66, 200)),  input_shape=(65,320,3)))

    model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Convolution2D(64,3,3,activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model            

def LeNet():
    
    model = Sequential()

    # Normalize image
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))

    # Cropping Layer
    
    # 50 rows pixels from the top of the image
    # 20 rows pixels from the bottom of the image
    # 0 columns of pixels from the left of the image
    # 0 columns of pixels from the right of the image

    #model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))


    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

def commaai():
    ch, row, col = 3, 160, 320  # camera format

    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
                     #  input_shape=(ch, row, col),
                     #  output_shape=(ch, row, col)))
                     input_shape=(row, col, ch),
                     output_shape=(row, col, ch)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mse")

    return model
    
# Choose model

# input_shape = (64,64,3)
# input_shape = (160,320,3)
# model = Nvidia()
# model = commaai()
# model = LeNet()
model = CNN()


def get_callbacks():
    # checkpoint = ModelCheckpoint(
    #     "checkpoints/model-{val_loss:.4f}.h5",
    #     monitor='val_loss', verbose=1, save_weights_only=True,
    #     save_best_only=True)

    # tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
    #                           write_graph=True, write_images=False)

    # return [checkpoint, tensorboard]

    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=1, mode='auto')
    # return [earlystopping, checkpoint]
    return [earlystopping]
    
adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

restart=True
if os.path.isfile(model_json) and restart:
    try:
        with open(model_json) as jfile:
            model = model_from_json(json.load(jfile))
            model.load_weights(model_weights)    
        print('loading trained model ...')
    except Exception as e:
        print('Unable to load model', model_name, ':', e)
        raise    

model.compile(optimizer=adam, loss='mse')

# Number of epoches
nb_epoch = 20
history = model.fit_generator(train_generator, samples_per_epoch=20000, nb_epoch=nb_epoch, validation_data=(X_validation,Y_validation),callbacks=get_callbacks(),verbose=1)

#callbacks=get_callbacks()               

                    
json_string = model.to_json()

print('Save the model')

try:
    os.remove(model_json)
    os.remove(model_weights)
except OSError:
    pass   

with open(model_json, 'w') as outfile:
    json.dump(json_string, outfile)
    model.save_weights(model_weights)



# serialize model to JSON
model_json = model.to_json()
with open("model_test.json", "w") as json_file:
    json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model_test.h5")
    
print("Saved model to disk")



print('Done')
