from sklearn.cluster import KMeans
import cv2
import matplotlib.pyplot as plt
import numpy as np

class ColourDeterminer:
    """Segment images using chromaticity information in the CIE LAB colour-space"""
    def __init__(self):
        self.rgb = None
        self.lab = None
        self.clt = None

    def load(self, img_filename):
        image = cv2.imread(img_filename)
        self.rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.lab = cv2.cvtColor(self.rgb, cv2.COLOR_RGB2LAB)

    @property
    def ab(self):
        """Return only the chromaticity components a,b from the Lab image"""
        return self.lab[:, :, 1:3]

    def cluster_colours(self, n = 2):
        self.clt = KMeans(n_clusters = n)
        ab_flattened = self.ab.reshape(self.lab.shape[0] * self.lab.shape[1], 2)
        self.clt.fit(ab_flattened)

    def plot_lab_histograms(self):
        hist_l, hist_a, hist_b = [ cv2.calcHist([self.lab],[n],None,[256],[0,256]) for n in range(3) ]
        plt.semilogy(hist_l, color = "gray", label = "L")
        plt.semilogy(hist_a, color = "magenta", label = "a")
        plt.semilogy(hist_b, color = "green", label = "b")
        plt.title("L, a, b histograms")

    def plot_rgb_histograms(self):
        hist_r, hist_g, hist_b = [ cv2.calcHist([self.rgb],[n],None,[256],[0,256]) for n in range(3) ]
        plt.semilogy(hist_r, color = "red", label = "R")
        plt.semilogy(hist_g, color = "green", label = "G")
        plt.semilogy(hist_b, color = "blue", label = "B")
        plt.title("R, G, B histograms")
    
    @property
    def colour_centres_ab(self):
        return self.clt.cluster_centers_

    def associate_to_cluster(self):
        """Return an image in which every point gives the index of the cluster whose centre is closest
        to the colour of the pixel in A-B coordinates"""
        clusters = self.colour_centres_ab
        n_clusters = len(clusters)
        ab = self.ab
        dst_sq = np.zeros((self.lab.shape[0], self.lab.shape[1], n_clusters), dtype=np.uint32)
        for n in range(n_clusters):
            dst_sq[:, :, n] = np.square(ab[:, :, 0] - clusters[n][0]) + np.square(ab[:, :, 1] - clusters[n][1])
        pixel_cluster = np.argmin(dst_sq, 2)
        return np.uint8(pixel_cluster) if n_clusters <= 255 else pixel_cluster

    @staticmethod
    def colour_as_lab(r, g, b):
        rgb = np.uint8([[[r, g, b]]])
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        return lab[0, 0, :]
