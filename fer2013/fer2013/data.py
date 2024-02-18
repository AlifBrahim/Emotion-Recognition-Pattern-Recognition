import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load the data
data = pd.read_csv('fer2013.csv')

# Convert pixels into numpy arrays
data['pixels'] = data['pixels'].apply(lambda pixel_sequence: np.fromstring(pixel_sequence, sep=' '))

# Reshape the pixels into 48x48 arrays (or whatever the correct dimensions are)
data['pixels'] = data['pixels'].apply(lambda pixel_array: pixel_array.reshape(48, 48, 1))

# Convert emotion labels to integers
le = LabelEncoder()
data['emotion'] = le.fit_transform(data['emotion'])

# Split the data into train and test sets
train = data[data['Usage'] == 'Training']
X_test = np.stack(test['pixels']) # Unresolved reference 'test'
y_test = test['emotion']

# Load the model
model = load_model('_mini_XCEPTION.102-0.66.hdf5')

# Get the model's predictions
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Get the confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Print the confusion matrix
print(cm)
