import numpy as np
import csv

class Searcher:

    def search(self, queryFeatures, limit = 11):
        results = {}
  		# open the index file for reading
        with open(self.indexPath) as f:
  		# initialize the CSV reader
            reader = csv.reader(f)
            # loop over the rows in the index
            for row in reader:
               
                x = row[0]
                row.pop(0)
                # row.pop(len(row) - 1)
                features = [float(x) for x in row]
                d = self.distance(features, queryFeatures)
                results[x] = d
            # close the reader
            f.close()

        results = sorted([(v, k) for (k, v) in results.items()])
        # return our (limited) results
        return results[:limit]

    def distance(self, A, B):
        A = np.array(A)
        B = np.array(B)
        # compute the Euclidean distance
        return np.linalg.norm(A-B)

    def __init__(self, indexPath):
  	# store our index path
  	    self.indexPath = indexPath

    
