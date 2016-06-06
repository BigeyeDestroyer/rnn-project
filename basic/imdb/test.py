from common.data import *
import numpy

path = get_dataset_file('imdb.pkl')

print path

train_set, valid_set, test_set = load_imdb()
print len(train_set[0])
print len(valid_set[0])
print len(test_set[0])
print train_set[0][0]
print train_set[1][0]

a = []
for sentence in train_set[0]:
    a.append(len(sentence))

print min(a)
print max(a)
print numpy.median(a)
print numpy.mean(a)