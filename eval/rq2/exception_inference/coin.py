from sklearn import metrics
import csv
import random

lbls = []
pos = 0

with open('/home/icse2022_artifact/data/methods2test_star/train.csv') as f:
    a = csv.reader(f)
    for row in a:
        try:
            lbls.append(int(row[0]))
        except ValueError:
            continue

pos = len([l for l in lbls if l])
tot = len(lbls)
pos_weight = pos/tot

test_lbls = []
with open('/home/icse2022_artifact/data/methods2test_star/test.csv') as f:
    a = csv.reader(f)
    for row in a:
        try:
            test_lbls.append(int(row[0]))
        except ValueError:
            continue


acc, precision, recall = 0, 0, 0
for i in range(0, 100):
    coin_preds = []
    for i in range(0, len(test_lbls)):
        r = random.random()
        if r <= pos_weight:
            if test_lbls[i] == 1: acc += 1
            coin_preds += [1]
        else:
            if test_lbls[i] == 0: acc += 1
            coin_preds += [0]


    precision += metrics.precision_score(test_lbls, coin_preds)
    recall += metrics.recall_score(test_lbls, coin_preds)

precision = round(precision / 100, 2)
recall = round(recall / 100, 2)
acc = round(acc/len(test_lbls), 2)
f1 =  round(2*((precision*recall)/(precision+recall)), 2)
print(f"accuracy: {acc}%")
print("precision:", precision)
print("recall:", recall)
print("f1:", f1)


