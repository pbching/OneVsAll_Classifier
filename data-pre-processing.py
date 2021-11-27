import underthesea
import json
import random

# with open("classification\Bất động sản\content.json", 'r', encoding="utf-8") as json_file:
#     data = json.load(json_file)
#     print(data)
#     with open("data.txt", 'w', encoding="utf-8") as f:
#         for i in range (1000):
#             json.dump(data[i]['message'] + data[i]['feature'] + '\n', f, ensure_ascii=False)

list = ['Bất động sản','Chính trị', 'Công nghệ', 'Đối ngoại', 'Đời sống', 'Du lịch', 'Giải trí', 'Giáo dục', 
'Khoa học', 'Kinh tế', 'Pháp luật', 'Quân sự', 'Thể thao', 'Văn hóa', 'Xã hội']

for i in list:
    with open("classification\\" + i + "\\content.json", 'r', encoding="utf-8") as json_file:
        data = json.load(json_file)
    l = len(data) - 1
    # '_'.join(i.strip())
    for j in range (1500):
        x = random.randint(1, l)
        with open("training_data\\" + str(i) + "\\" + str(j) + ".txt", 'w', encoding="utf-8") as f:
            json.dump(data[x]['message'] + data[x]['feature'], f, ensure_ascii=False)

