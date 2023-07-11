import json
import numpy as np
import matplotlib.pyplot as plt

def load_ground_truth(path):
    with open(path) as json_file:
        gt = json.load(json_file)
        # Skippa tillfälligt för en bild gick sönder
        gt.pop('005350', None)
        for f in gt.keys():
            gt[f] = np.array(gt[f])
        return gt

ground_truth = load_ground_truth("/mnt/ZOD/ground_truth.json")
frame_ids = ["000001"]

for frame_id in frame_ids:
    gt = ground_truth[frame_id]
    xs, ys, zs = [],[],[]
    for i in range(0,len(gt),3):
        coord = gt[i:i+3]
        xs.append(coord[0])
        ys.append(coord[1])
        zs.append(coord[2])
        
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xs, ys, zs, marker="o")

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    plt.savefig("plot.png")