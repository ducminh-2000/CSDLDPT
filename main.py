import glob
import os
import cv2
from ColorDescriptor import ColorDescriptor
from Search import Searcher
from canny_custom import my_canny
import numpy as np


cd = ColorDescriptor((8, 12, 3))
query = cv2.imread("dataset_2/mango_tree/image0.jpg")
image = cv2.imread("dataset_2/mango_tree/image0.jpg",0)

cv2.imwrite('test/test_query.jpg',my_canny(image, min_val=100, max_val=200))
features_color = cd.describe(query)
features_edges = cd.describe(cv2.imread("test/test_query.jpg"))
features = np.concatenate((features_edges,features_color))

# perform the search
searcher = Searcher("data.csv")
results = searcher.search(features)


# display the query
# cv2.imshow("Query", query)
rs = {}
for inputFolder in glob.glob("dataset_2/*"):
    inputFolder = inputFolder.split('/')[1] 
    i = 0   
    for x in results:
        temp = x[1].split('_')[0] + '_' +x[1].split('_')[1]
        # print(temp)
        if(temp.__eq__(inputFolder)):
            i += 1
        rs[inputFolder] = i

rs = sorted([(v, k) for (k, v) in rs.items()])
print(rs[len(rs)-1])

    
