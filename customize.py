import os
import imageio
from sklearn.cluster import KMeans
import numpy as np
import pickle
from PIL import Image
import json
import spacy
import pandas as pd
from random import choice

path = './img-apple-64'
dirs = os.listdir(path)

clt = KMeans()
from scipy.spatial import distance
import codecs
with open('dc_dict.pickle', 'rb') as handle:
	dominant_color_dict = pickle.load(handle)

keywords = ['cute', 'smile']

def load_emojis():
	rows = []
	with open('./emojis.json') as f:
		for emoji in json.loads(f.read()):
			rows.append([emoji['name'], emoji['unicode'], ' '.join(emoji['keywords'])])    
	return np.array(rows)

emojis = load_emojis()
nlp = spacy.load('en_core_web_sm')
from tqdm import tqdm

with open('./glove.6B/glove.6B.100d.txt', 'r') as f:
	for line in tqdm(f, total=400000):
		parts = line.split()
		word = parts[0]
		vec = np.array([float(v) for v in parts[1:]], dtype='f')
		nlp.vocab.set_vector(word, vec)

docs = [nlp(str(keywords)) for _, _, keywords in tqdm(emojis)]
doc_vectors = np.array([doc.vector for doc in docs])
from numpy import dot
from numpy.linalg import norm

def most_similar(vectors, vec):
	cosine = lambda v1, v2: dot(v1, v2) / (norm(v1) * norm(v2))
	dst = np.dot(vectors, vec) / (norm(vectors) * norm(vec))
	return np.argsort(-dst)

def query(v, most_n=5):
	ids = most_similar(doc_vectors, v)[:most_n]
	return ids

def add_custom(dominant_color_dict):
	weight_more = set([])
	for keyword in keywords:
		v = nlp(keyword.lower()).vector
		ids = query(v)

		for ind in ids:
			unicode_n = emojis[ind][1].replace('U+','').split()
			name = '-'.join(unicode_n).lower()
			try:
				im = Image.open(path+'/'+name+'.png')
			except:
				continue
			m = np.asarray(im)
			if m.shape != (64,64,4):
				continue
			data = []
			for i in range(m.shape[0]-1):
				for j in range(m.shape[1]-1):
					if m[i][j][3] == 255:
						data.append(m[i][j][:-1])
			data = np.array(data)
			try:
				clt.fit(data)
				(values, counts) = np.unique(clt.labels_, return_counts=True)
				index = np.argmax(counts)
				dominant_color_dict[name] = clt.cluster_centers_[values[index]]
				weight_more.add(name)
				print(name)
			except:
				continue

	return dominant_color_dict, weight_more

dominant_color_dict, weight_more = add_custom(dominant_color_dict)

# outfile = codecs.open('test.html','w','utf-8')
vector = []
reverse_dict = {}
ind = 0
weight_more_ind = set([])
for key in dominant_color_dict:
	vector.append(dominant_color_dict[key])
	if key in weight_more:
		weight_more_ind.add(ind)
	reverse_dict[ind] = key
	ind += 1
vector = np.array(vector)


from PIL import Image
basewidth = 100
img = Image.open("/Users/chen/Desktop/gavin.jpg")
from PIL import ImageEnhance
wpercent = (basewidth/float(img.size[0]))
hsize = int((float(img.size[1])*float(wpercent)))
img = img.resize((basewidth,hsize), Image.ANTIALIAS)
img.save('sompic.jpg')
# test_image = imageio.imread('/Users/chen/Desktop/test1.jpeg')
# test_image = imageio.imread("/Users/chen/Desktop/nf48l.jpg")

test_image = Image.open("sompic.jpg")
m = np.asarray(test_image)
print(m.shape)
clt = KMeans(n_clusters=100)
clt.fit(m.reshape((-1,3)))
t = clt.cluster_centers_
print(t.shape)
print(clt.labels_.shape)
label = clt.labels_.reshape((m.shape[0], m.shape[1]))
distances = distance.cdist(t, vector, "euclidean")
min_index = np.argsort(distances, axis=1)
import random
print(min_index.shape)
new_im = Image.new('RGB', (m.shape[1]*64, m.shape[0]*64), color=(255,255,255))
wrong = set([])
for i in range(m.shape[0]):
	for j in range(m.shape[1]):
		mix = []
		# for k in range(5):
		# 	mix.append(reverse_dict[min_index[label[i][j]][k]])
		for w in weight_more_ind:
			if w in min_index[label[i][j]][:50]:
				mix.append(reverse_dict[w])
		x = max(len(mix), 5)
		for k in range(x):
			mix.append(reverse_dict[min_index[label[i][j]][k]])
		s = choice(mix)
		# s = reverse_dict[min_index[label[i][j]][random.randint(0,5)]]

		# for w in weight_more_ind:
		# 	if w in min_index[label[i][j]][:150]:
		# 		s = reverse_dict[w]
		# s = reverse_dict[min_index[label[i][j]][random.randint(0,5)]]
		im = Image.open(path+'/'+s+'.png')
		try:
			new_im.paste(im, (j*64, i*64), im)
		except:
			wrong.add(s)
print(wrong)
new_im.save("gavin-100.jpg")


print("Done!")





