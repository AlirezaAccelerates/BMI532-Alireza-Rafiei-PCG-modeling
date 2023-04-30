import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2DTranspose, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

# Define the input shape
img_rows = 32
img_cols = 240
channels = 1
img_shape = (img_rows, img_cols, channels)

# Define the generator model
def build_generator():

    model = Sequential()

    model.add(Dense(128 * 8 * 60, activation="relu", input_dim=100))
    model.add(Reshape((8, 60, 128)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2DTranspose(128, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2DTranspose(64, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2DTranspose(64, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2DTranspose(channels, kernel_size=4, padding="same"))
    model.add(Activation("tanh"))

    noise = Input(shape=(100,))
    img = model(noise)

    return Model(noise, img)

# Define the discriminator model
def build_discriminator():

    model = Sequential()

    model.add(Conv2D(128, kernel_size=4, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=4, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=4, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=4, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(2, activation='softmax'))

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)

# Define the combined generator-discriminator model, for updating the generator
def build_combined(generator, discriminator):

    # Freeze the discriminator's weights during generator training
    discriminator.trainable = True

    # Sample noise and generate a batch of fake images
    noise = Input(shape=(100,))
    img = generator(noise)

    # Determine the validity of the fake images
    validity = discriminator(img)

    # The combined model (stacked generator and discriminator)
    # Trains the generator to fool the discriminator
    combined = Model

    
    def build_combined(generator, discriminator):

    # Freeze the discriminator's weights during generator training
    discriminator.trainable = True

    # Sample noise and generate a batch of fake images
    noise = Input(shape=(100,))
    img = generator(noise)

    # Determine the validity of the fake images
    validity = discriminator(img)

    # The combined model (stacked generator and discriminator)
    # Trains the generator to fool the discriminator
    combined = Model(noise, validity)
    combined.compile(loss='binary_crossentropy', optimizer=Adam())

    return combined

# Build the generator
generator = build_generator()

# Build and compile the discriminator
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

# Build and compile the combined model
combined = build_combined(generator, discriminator)


#(X_train - np.min(x_train)) / (np.max(x_train) - np.mean(x_train))
x_train = (X_train - np.min(X_train)) / (np.max(X_train) - np.mean(X_train))
#X_train = np.expand_dims(X_train, axis=3)

# Define training parameters
epochs = 1000
batch_size = 32
save_interval = 1000

# Define real and fake labels for the discriminator
real = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))



# Train the GAN
for epoch in range(epochs):

    # Train the discriminator
    # Select a random batch of real images
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_imgs = x_train[idx]
    real_label = y_train[idx]
    
    fake_label = np.random.randint(0, 2, size=(batch_size, 2))
    

    # Generate a batch of fake images
    noise = np.random.normal(0, 1, (batch_size, 100))
    fake_imgs = generator.predict(noise)

    # Train the discriminator on real and fake images
    d_loss_real = discriminator.train_on_batch(real_imgs, real_label)
    d_loss_fake = discriminator.train_on_batch(fake_imgs, fake_label)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    # Train the generator
    # Generate a batch of noise samples
    noise = np.random.normal(0, 1, (batch_size, 100))

    # Train the generator to fool the discriminator
    g_loss = combined.train_on_batch(noise, real_label)

    # Print the progress
    print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
