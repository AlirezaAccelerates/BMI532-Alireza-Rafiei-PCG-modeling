from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(32, 240, 1))
x = layers.Rescaling(1./255)(inputs)
x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
x = layers.Conv2D(filters=16, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Dropout(0.2)(x)

x = layers.Flatten()(x)
x = layers.Dense(32, activation="relu")(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(3, activation="softmax")(x)
model = keras.Model(inputs=inputs, outputs=outputs)

model.summary()


model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
hist = model.fit(X_train, y_train, batch_size=32, epochs=60,  shuffle=True)


History = hist.history
losses = History['loss']
accuracies = History['accuracy']

model.save('CNN.h5')
