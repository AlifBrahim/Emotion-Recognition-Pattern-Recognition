import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score , ConfusionMatrixDisplay
from skimage import color, feature, exposure
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV
import joblib

# load the data
data_train = pd.read_csv('fer2013/fer2013/fer2013.csv')

# make the dataset to an array of images of pixels
image_array = []
for i, row in enumerate(data_train.index):
    image = np.fromstring(data_train.loc[row, 'pixels'], dtype=int, sep=' ')
    image_array.append(image.flatten())

labels = np.array(data_train['emotion']).tolist()
flat_images = np.array(image_array)
target = np.array(labels)

# normalization
flat_images = flat_images / 255

df = pd.DataFrame(flat_images)
df['target'] = target

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

def extract_hog_features(image):
    gray_image = image
    hog_features, hog_image = feature.hog(gray_image, visualize=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return hog_features , hog_image_rescaled

hog_features_list = []
hog_images = []
for index, row in X.iterrows():
    image_pixels = row.values.reshape(48, 48)
    hog_features ,hog_image = extract_hog_features(image_pixels)
    hog_features_list.append(hog_features)
    hog_images.append(hog_image)

hog_features_array = np.array(hog_features_list)

X_train, X_test, y_train, y_test = train_test_split(hog_features_array, y, test_size=0.2, random_state=42)

# Binarize the output
lb = LabelBinarizer()
y_train_bin = lb.fit_transform(y_train)
y_test_bin = lb.transform(y_test)
'''
# Define the parameter grid for SVM
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],  
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
    'kernel': ['rbf']
}  

# Create a base model
svc = svm.SVC(probability=True)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=svc, param_grid=param_grid, cv=3, n_jobs=-1, verbose=3)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_


'''
print("Best parameters: C=0.1, gamma=0.1, kernel='rbf', probability=True ")
# Train the model using the best parameters
model = svm.SVC(C=0.1, gamma=0.1, kernel='rbf', probability=True)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'svm_model.pkl')

# Predict the labels
y_predict = model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_predict)
print("Accuracy: ", accuracy)

# Display the confusion matrix
con = confusion_matrix(y_test, y_predict)
print("Confusion Matrix:")
print(con)

# Display the confusion matrix
classes = lb.classes_
disp = ConfusionMatrixDisplay(confusion_matrix=con, display_labels=classes)
disp.plot(cmap='Reds')
plt.title('the Confusion Matrix')
plt.show()

# Predict probabilities for each class
y_score = model.decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

num_classes = len(classes)

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
