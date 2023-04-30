from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Activation, Dropout, Flatten, Dense

# Define the DeepConvNet model
model = Sequential()

# Block 1
model.add(Conv2D(25, kernel_size=(1, 5), strides=(1, 1), input_shape=(32, 240, 1), padding='valid', activation=None, use_bias=True))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid'))
model.add(Dropout(0.5))

# Block 2
model.add(Conv2D(50, kernel_size=(1, 5), strides=(1, 1), padding='valid', activation=None, use_bias=True))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid'))
model.add(Dropout(0.5))

# Block 3
model.add(Conv2D(100, kernel_size=(1, 5), strides=(1, 1), padding='valid', activation=None, use_bias=True))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid'))
model.add(Dropout(0.5))

# Block 4
model.add(Conv2D(200, kernel_size=(1, 5), strides=(1, 1), padding='valid', activation=None, use_bias=True))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid'))
model.add(Dropout(0.5))

# Block 5
model.add(Conv2D(200, kernel_size=(1, 5), strides=(1, 1), padding='valid', activation=None, use_bias=True))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid'))
model.add(Dropout(0.5))

# Flatten the output of the last convolutional layer
model.add(Flatten())

# Fully connected layers
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
model.summary()

model.fit(data_all_train, label_all_train, batch_size=32, epochs=100,  shuffle=True)

model.save('DeepConvNet.h5')
