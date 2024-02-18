import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score , ConfusionMatrixDisplay
from skimage import color, feature, exposure
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve, roc_auc_score, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib

# load the data
data_train = pd.read_csv('fer2013/fer2013/fer2013.csv')
data_train.head()

# make the datset to an array of images of pixels 
image_array =[]
for i, row in enumerate(data_train.index):
        image = np.fromstring(data_train.loc[row, 'pixels'], dtype=int, sep=' ')
        image_array.append(image.flatten())

image_array
image_array[0].shape
lables = np.array(data_train['emotion']).tolist()
lables
flat_images = np.array(image_array)
target = np.array(lables)

# normalization
flat_images = flat_images / 255
for i in range(5):
    plt.figure(figsize=(1, 2))
    plt.imshow(image_array[i].reshape(48,48), cmap=plt.cm.gray)
    # remove ticks
    plt.xticks([])
    plt.yticks([])
    plt.show()
df = pd.DataFrame(flat_images)
df['target']=target

# names of classes
df['target'].unique()
count0= len(df[df['target'] == 0])
count1= len(df[df['target'] == 1])
count2= len(df[df['target'] == 2])
count3= len(df[df['target'] == 3])
count4= len(df[df['target'] == 4])
count5= len(df[df['target'] == 5])
count6= len(df[df['target'] == 6])
print('number of Angry images: ',count0)
print('number of Disgust images: ',count1)
print('number of Fear images: ',count2)
print('number of Happy images: ',count3)
print('number of Sad images: ',count4)
print('number of Surprise images: ',count5)
print('number of Neutral images: ',count6)

X = df.iloc[:,:-1]
y = df.iloc[:,-1]
def extract_hog_features(image):
    gray_image = image
    # Calculate HOG features
    hog_features, hog_image = feature.hog(gray_image, visualize=True)
    
    # Enhance the contrast of the HOG image for better visualization
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return hog_features , hog_image_rescaled
hog_features_list = []
hog_images=[]
for index, row in X.iterrows():
    image_pixels = row.values.reshape(48, 48) 
    hog_features ,hog_image = extract_hog_features(image_pixels)
    hog_features_list.append(hog_features)
    hog_images.append(hog_image)

hog_features_array = np.array(hog_features_list)
for i in range(2,7):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2,1), sharex=True, sharey=True)
    ax1.axis('off')
    ax1.imshow(image_array[i].reshape(48,48), cmap=plt.cm.gray)
    ax2.axis('off')
    ax2.imshow(hog_images[i], cmap=plt.cm.gray)
    plt.show()
    
X_train, X_test, y_train, y_test = train_test_split(hog_features_array, y, test_size=0.2, random_state=42)

# Define the parameter grid
'''
param_grid = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'C': np.logspace(-4, 4, 20),
    'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
    'max_iter': [100000, 200000, 250000, 500000]
}

# Create a base model
logistic = LogisticRegression()

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=logistic, param_grid=param_grid, cv=3, n_jobs=-1, verbose=3)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)
'''

# Get the best parameters
best_params = {
    'C': 0.615848211066026,
    'max_iter': 200000,  # or 250000 or 500000
    'penalty': 'l2',
    'solver': 'sag'  # or 'saga'
}

print("Best parameters: ", best_params)

# Train the model using the best parameters
model = LogisticRegression(**best_params)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'logistic_regression_model.pkl')

y_predict = model.predict(X_test)
y_predict
y_test
test_list = y_test.to_list()
accuracy = accuracy_score(test_list,y_predict)
print(accuracy)
con = confusion_matrix(y_test, y_predict)
print("Confusion Matrix:")
print(con)
# classes = model.classes_
# it will return 0 , 1 , 2 , 3 , 4   
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
disp = ConfusionMatrixDisplay(confusion_matrix=con, display_labels=classes)
disp.plot(cmap='Reds')
plt.title('the Confusion Matrix')
plt.show()
y_test_bin = label_binarize(y_test, classes=np.unique(y_train))
# Predict probabilities for each class
y_score = model.decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

num_classes = len(np.unique(y_train))

for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
