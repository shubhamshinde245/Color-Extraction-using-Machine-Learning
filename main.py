import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
import time

def get_image(path):
    image = cv2.imread(path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    print("Resolution: {}".format(image.shape))
    print("Extracting colour scheme, please wait.")
    plt.imshow(image)
    return image

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]),int(color[1]),int(color[2]))

def get_colors(image,number_of_colors,show_chart):
    modified_image = cv2.resize(image,(600,400),interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1],3)

    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)

    counts = Counter(labels)
    center_colors = clf.cluster_centers_
    order_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(order_colors[i]) for i in counts.keys()]
    rgb_colors = [order_colors[i] for i in counts.keys()]

    if(show_chart):
        plt.figure(figsize = (8,6))
        plt.pie(counts.values(),labels = hex_colors, colors = hex_colors,autopct='%1.2f%%')
    t_stop = time.process_time()  
    print("{} seconds required to execute.".format(t_stop-t_start))
    return rgb_colors

path = input("Enter image Location :")
t_start = time.process_time()
get_colors(get_image(path),8,True)
