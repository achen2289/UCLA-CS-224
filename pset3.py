import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import os

def main():
    np.random.seed(0)

    with open("pset 3/q4.data", "r") as f:
        data = f.read().strip("\n")

        x, y = [], []
        for line in data.split("\n")[1:]:
            line_clean = line.split()
            x.append(line_clean[2:])
            y.append(line_clean[:2])

    pca = PCA(n_components=2)
    pca.fit(x)
    reduced = pca.transform(x)
    print (reduced.shape)
    # print (pca.get_feature_names_out)

    pca1, pca2 = [l[0] for l in reduced], [l[1] for l in reduced]
    
    plt.scatter(pca2, pca1)
    plt.xlabel("pc2")
    plt.ylabel("pc1")
    plt.title("PC1 vs PC2")
    plt.show()




if __name__ == "__main__":
    main()