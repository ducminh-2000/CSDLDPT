import cv2
from ColorDescriptor import ColorDescriptor
from Search import Searcher


cd = ColorDescriptor((8, 12, 3))
query = cv2.imread("bien.jpg")
features = cd.describe(query)
# perform the search
searcher = Searcher("file.csv")
results = searcher.search(features)
# display the query
cv2.imshow("Query", query)
print(results)
# loop over the results
# for (score, resultID) in results:
# 	# load the result image and display it
# 	result = cv2.imread(resultID)
# 	cv2.imshow("Result", result)
#   cv2.waitKey()
    
