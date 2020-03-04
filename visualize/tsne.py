
from sklearn.manifold import TSNE
import numpy as np
from typing import List


import matplotlib.pyplot as plt
import matplotlib.cm as cm
# % matplotlib inline
#
# words = ["aaa", "bbb", "ccc", "d", "sdf", "a"]
# vectors =np.random.rand(6,100)
#
# words_ak = []
# embeddings_ak = []
# for i, word in enumerate(words):
#     embeddings_ak.append(vectors[i])
#     words_ak.append(word)
#
# tsne_ak_2d = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3500, random_state=32)
# embeddings_ak_2d = tsne_ak_2d.fit_transform(embeddings_ak)




def tsne_plot_2d(label, embeddings, words: List=[], a:float=1, file_name:str = None):
    plt.figure(figsize=(16, 9))
    colors = cm.rainbow(np.linspace(0, 1, 1))
    x = embeddings[:, 0]
    y = embeddings[:, 1]
    plt.scatter(x, y, c=colors, alpha=a, label=label)
    for i, word in enumerate(words):
        plt.annotate(word, alpha=0.3, xy=(x[i], y[i]), xytext=(5, 2),
                     textcoords='offset points', ha='right', va='bottom', size=10)
    plt.legend(loc=4)
    plt.grid(True)
    plt.savefig(f"{file_name}.png", format='png', dpi=150, bbox_inches='tight')
    # plt.show()


# tsne_plot_2d('Anna Karenina by Leo Tolstoy', embeddings_ak_2d, a=0.1, words=words_ak)
