# Keras Import for Model

from keras.models import Input, Model
from keras.layers import TimeDistributed, LSTM
from keras.layers import ConvLSTM2D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, LeakyReLU, BatchNormalization
from keras.layers import Dense, Flatten, GlobalMaxPooling2D
from keras.layers import MaxPooling3D
from keras.layers import concatenate

from keras.optimizers import Adam

# Residual Block for Feature Extraction

def res_block(model, filters):
  start_block = model
  model = Conv2D(filters=filters, kernel_size = 3, padding='same')(model)
  model = BatchNormalization(momentum=0.9)(model)
  model = LeakyReLU(0.2)(model)
  return concatenate([start_block, model])

# Deep Crime Detection Model

def create_model():
    input_layer = Input(shape=(frames, Width, Height,3))
    model =  ConvLSTM2D(32,3,padding='same',return_sequences=False)(input_layer)
    model = BatchNormalization(momentum=0.9)(model)
    model = LeakyReLU(0.2)(model)
    filters = 64
    for _ in range(6):
      model = res_block(model, filters)
      try:
        model = MaxPooling3D((2,2,2))(model)
      except:
        model = MaxPooling2D((2,2))(model)
      if filters < 512:
        filters *= 2
    model = Flatten()(model)
    model = Dense(classes, activation='softmax')(model)
    model = Model(input_layer, model)
    model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return model
  
  classifier = create_model()
  
  # Path to save the model
  
  classifier.save("my_model.h5")
  
  
