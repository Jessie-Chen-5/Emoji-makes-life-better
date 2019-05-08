import os
from sklearn.cluster import KMeans
import numpy as np
import pickle
from PIL import Image

path = './img-apple-64'
dirs = os.listdir(path)
count = 0
clt = KMeans(n_clusters=3)
dominant_color_dict = {}
for filename in dirs:
	if len(filename) == 9 and filename[-3:] == 'png':
		im = Image.open(path+'/'+filename)
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
			ind = np.argmax(counts)
			dominant_color_dict[filename[:-4]] = clt.cluster_centers_[values[ind]]
			count += 1
		except:
			continue

with open('dc_dict.pickle', 'wb') as handle:
	pickle.dump(dominant_color_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)