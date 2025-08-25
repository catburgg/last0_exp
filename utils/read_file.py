
import gzip
import pickle
import numpy as np

# with gzip.open("/gpfs/0607-cluster/guchenyang/Data/R1LITE/Compressed/0822/0.pkl.gz", 'rb') as f:
#     data = pickle.load(f)

# print(data[0])


episode = np.load("/gpfs/0607-cluster/guchenyang/Data/R1LITE/Processed/0821_pick_place_keyframe/51.npy", allow_pickle=True)

print(episode[0])
print(len(episode))
