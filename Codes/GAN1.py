from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from keras.models import Sequential, Model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2D, Conv2DTranspose, BatchNormalization, LeakyReLU,Concatenate
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.datasets import mnist
import numpy as np

# Set seed for reproducibility
np.random.seed(42)

# Load data and normalize
x_train = S_DB_all
y_train = murmurs_sec
x_train = (X_train - np.min(X_train)) / (np.max(X_train) - np.mean(X_train))
#x_train = np.expand_dims(x_train, axis=3)

# Define generator
def build_generator():
    noise_shape = (128,)
    
    model = Sequential()

    model.add(Dense(128 * 8 * 30, activation="relu", input_shape=noise_shape))
    model.add(Reshape((8, 30, 128)))
    model.add(Conv2DTranspose(16, kernel_size=4, strides=2, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(32, kernel_size=4, strides=2, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(64, kernel_size=4, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(1, kernel_size=4, strides=(1,2), padding="same", activation="tanh"))

    noise = Input(shape=noise_shape)
    img = model(noise)

    return Model(noise, img)

# Define discriminator
def build_discriminator():
    img_shape = (32, 240, 1)
    
    model = Sequential()

    model.add(Conv2D(16, kernel_size=4, strides=2, input_shape=img_shape, padding="same", activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=4, strides=2, padding="same", activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=4, strides=2, padding="same", activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=4, strides=2, padding="same", activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=4, strides=2, padding="same", activation=LeakyReLU(alpha=0.2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)
  
  
discriminator = build_discriminator()
generator = build_generator()

# Compile the discriminator
discriminator.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

# Set the discriminator's weights to be non-trainable
discriminator.trainable = True

# Combine the generator and discriminator into a single model
gan_input = Input(shape=(128,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
gan = Model(inputs=gan_input, outputs=gan_output)

# Compile the GAN
gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

# Define the training loop
def train(epochs, batch_size):
    # Calculate the number of batches per epoch
    batch_count = x_train.shape[0] // batch_size

    for epoch in range(1, epochs+1):
        for batch in range(batch_count):
            # Generate a batch of fake images
            noise = np.random.normal(0, 1, size=(batch_size, 128))
            fake_images = generator.predict(noise)

            # Get a batch of real images
            real_images = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]

            # Combine the real and fake images into a single batch
            X = np.concatenate([real_images, fake_images])

            # Create the labels for the discriminator
            y_discriminator = np.zeros(2*batch_size)
            y_discriminator[:batch_size] = 0.9

        # Train the discriminator
        discriminator_loss = discriminator.train_on_batch(X, y_discriminator)

        # Generate a new batch of noise
        noise = np.random.normal(0, 1, size=(batch_size, 128))

        # Create labels for the generator
        y_generator = np.ones(batch_size)

        # Train the generator
        gan_loss = gan.train_on_batch(noise, y_generator)


        print(f'Epoch {epoch}/{epochs} Discriminator Loss: {discriminator_loss[0]} Generator Loss: {gan_loss}')

train(epochs=100, batch_size=64)
