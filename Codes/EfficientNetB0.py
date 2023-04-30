import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0

# Load the pre-trained EfficientNetB0 model without the top layer
efficientnetb0_base = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(32, 240, 3))

# Freeze the base model's layers to prevent further training
for layer in efficientnetb0_base.layers:
    layer.trainable = False

# Add custom top layers for your specific problem
x = efficientnetb0_base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dropout(0.2)(x)
predictions = layers.Dense(3, activation='softmax')(x)

# Create the new model combining the base and top layers
model = models.Model(inputs=efficientnetb0_base.input, outputs=predictions)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=tf.constant(0.001).numpy()),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train2, y_train , batch_size=32, epochs=100,  shuffle=True)

model.save('EfficientNetB0.h5')
