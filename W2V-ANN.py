import json
import pickle
import random
import numpy as np
from gensim.models import doc2vec
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from pyvi import ViTokenizer
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
import os
import logging # Setting up the loggings to monitor gensim

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
# os.environ[''] = '2'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

X=[]
y=[]

random.seed(20)

# Đọc file từ dừng
with open("stopwords_vn.txt", encoding="utf-8") as f:
  stopwords = f.readlines()
stopwords = [x.strip().replace(" ", "_") for x in stopwords]

list = ['Bất động sản','Chính trị', 'Công nghệ', 'Du lịch', 'Đối ngoại', 'Giáo dục', 
'Khoa học', 'Kinh tế', 'Pháp luật', 'Quân sự']

# Đọc dữ liệu 
with open("classification/Thể thao/content.json", 'r', encoding="utf-8") as f:
    data1 = json.load(f)
    for j in range(10000):
        words = []
        contents = (data1[j]['message']+data1[j]['feature']).split()
        for word in contents:
            if word not in stopwords:
                words.append(word)
        tagged_document = doc2vec.TaggedDocument(words, [j])
        X.append(tagged_document)
        y.append('Thể thao')

for i in list:
    with open("classification//" + str(i) + "//content.json", 'r', encoding="utf-8") as f:
        data = json.load(f)
        l = len(data)-10
        for j in range(1000):
            x = random.randint(1, l)    
            words = []
            contents = (data[x]['message']+data[x]['feature']).split()
            for word in contents:
                if word not in stopwords:
                    words.append(word)
            tagged_document = doc2vec.TaggedDocument(words, [j])
            X.append(tagged_document)
            y.append('Khác')

# Train model Doc2Vec
# model = doc2vec.Doc2Vec(vector_size=300, min_count=2, epochs=40)
# model.build_vocab(X)
# model.train(X, total_examples=model.corpus_count, epochs=model.epochs)
# model.save("doc2vec.model")
# Transform tập dữ liệu theo Doc2Vec
model = doc2vec.Doc2Vec.load("doc2vec.model")
X_data_vectors = []
for x in X:
    vector = model.infer_vector(x.words)
    X_data_vectors.append(vector)

# with open('X_data.pickle', 'rb') as handle:
    # X_data_vectors = pickle.load(handle)
print(np.array(X_data_vectors).shape)
encoder = preprocessing.LabelEncoder()
y_data = encoder.fit_transform(y)

def train_model(classifier, X_data, y_data, n_epochs=3):
    X_train_val, X_test, y_train_val, y_test = train_test_split(X_data, y_data, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=42)
    classifier.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=n_epochs, batch_size=64,verbose=1)
    
    val_predictions = classifier.predict(X_val)
    test_predictions = classifier.predict(X_test)
    val_predictions = val_predictions.argmax(axis=-1)
    test_predictions = test_predictions.argmax(axis=-1)

    print("Validation accuracy: ", accuracy_score(val_predictions, y_val))
    print("Test accuracy: ", accuracy_score(test_predictions, y_test))

def createDNN():
    input_layer = keras.Input(shape=(300,))
    layer = layers.Dense(1024, activation='relu')(input_layer)
    layer = layers.Dense(1024, activation='relu')(layer)
    layer = layers.Dense(512, activation='relu')(layer)
    output_layer = layers.Dense(2, activation='softmax')(layer)
    
    classifier = models.Model(input_layer, output_layer)
    classifier.compile(optimizer=keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return classifier

DNN = createDNN()

train_model(classifier=DNN, X_data=X_data_vectors, y_data=y_data, n_epochs=30)
