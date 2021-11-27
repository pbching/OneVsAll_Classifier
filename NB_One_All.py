import datetime
import json
import random
import numpy as np
import time
import pickle
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
from pyvi import ViTokenizer
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

X=[]
y=[]
random.seed(20)

list = ['Bất động sản','Chính trị', 'Công nghệ', 'Du lịch', 'Đối ngoại', 'Giáo dục', 
'Khoa học', 'Kinh tế', 'Pháp luật', 'Quân sự']
with open("classification\Thể thao\content.json", 'r', encoding="utf-8") as f:
    data1 = json.load(f)
    for j in range(10000):
        X.append(data1[j]['message']+data1[j]['feature'])
        y.append('Thể thao')

for i in list:
    with open("classification\\" + str(i) + "\\content.json", 'r', encoding="utf-8") as f:
        data = json.load(f)
        l = len(data)-10
        for j in range(1000):
            x = random.randint(1, l)
            X.append(data[x]['message']+data[x]['feature'])
            y.append('Khác')


with open("stopwords_vn.txt", encoding="utf-8") as f:
  stopwords = f.readlines()
stopwords = [x.strip().replace(" ", "_") for x in stopwords]

tfidf_vector = TfidfVectorizer(stop_words=stopwords)

data_preprocessed = tfidf_vector.fit_transform(X, y)

# X_train, X_test, y_train, y_test = train_test_split(data_preprocessed, data_train.target, random_state=10, test_size=0.15)
X_train, X_test, y_train, y_test = train_test_split(data_preprocessed, y, test_size=0.15)

print("Training...")
time_start = time.time()

model = MultinomialNB()

model.fit(X_train, y_train)
print("Traning model Multinomial NB complete in: " + str(time.time()-time_start))

print("Testing...")
y_pred = model.predict(X_test)
print(f"Multinomial NB accuracy: {accuracy_score(y_test, y_pred)}")

new_doc =  "Trong bối cảnh Mohamed Salah vẫn chưa đạt thỏa thuận gia hạn hợp đồng với Liverpool, khi vướng mắc về mức lương, Real Madrid quyết định đánh tiếng với cầu thủ người Ai Cập. Giới truyền thông Tây Ban Nha đưa tin, Real Madrid đã liên hệ với đại diện của Salah để hỏi về các yêu cầu cá nhân. Hợp đồng của Salah với Liverpool còn thời hạn đến 2023 và anh yêu cầu mức lương 400.000 bảng. Đây là con số mà The Kop không thể đáp ứng. Real Madrid đưa ra đề nghị thấp hơn so với đòi hỏi của Salah. Đổi lại, anh được hưởng nhiều khoản đãi ngộ khác, bên cạnh việc khai thác hình ảnh cá nhân. Mùa này, Salah ghi 17 bàn và 6 pha kiến tạo sau 17 trận. Chính hiệu suất này khiến Chủ tịch Florentino Perez càng quyết tâm kéo anh về sân Bernabeu trong mùa hè 2022."

# Trước hết, cần thực hiện tách từ sử dụng pyvi
tokenized_new_doc = ViTokenizer.tokenize(new_doc)
# Cần đưa văn bản ở dạng mảng/vector
tokenized_new_doc = [tokenized_new_doc]
# Rồi sử dụng module model_rf_preprocess
input_data_preprocessed = tfidf_vector.transform(tokenized_new_doc)
print(input_data_preprocessed)
label = model.predict(input_data_preprocessed)
print(model.predict_proba(input_data_preprocessed))
print('label ', label)