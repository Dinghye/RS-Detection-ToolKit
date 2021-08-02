
import json
import os


"""
NOT SUPPOSED TO BE HERE!!!
PATCH OF "utils/ImageSplit.py"
"""

def save_file(path, item):

        # 先将字典对象转化为可写入文本的字符串
        item = json.dumps(item)

        try:
            if not os.path.exists(path):
                with open(path, "w", encoding='utf-8') as f:
                    f.write(item + ",\n")
                    print("^_^ write success")
            else:
                with open(path, "a", encoding='utf-8') as f:
                    f.write(item + ",\n")
                    print("^_^ write success")
        except Exception as e:
            print("write error==>", e)

json_file = '../dataset/splitMyDataset/annotations/val.json'
with open(json_file) as i:
    info = json.load(i)

for i in range(0,len(info['images'])):
    info['images'][i]['height'] = 1024
    info['images'][i]['width'] = 1024

save_file('../dataset/splitMyDataset/annotations/val_edit.json',info)



