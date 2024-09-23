
import tensorflow as tf
from keras import layers, models


from preprocess import X_train
from preprocess import y_train
from preprocess import num_classes
from preprocess import words
from preprocess import X_test
from preprocess import y_test

# Define the model
model = models.Sequential()
model.add(layers.Dense(128, input_shape=(len(words),), activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1, validation_data=(X_test, y_test))

# Save the model
model.save('chatbot_model.h5')