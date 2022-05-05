import numpy as np
import csv

class Searcher:

    def search(self, queryFeatures, limit = 10):
        results = {}
  		# open the index file for reading
        with open(self.indexPath) as f:
  		# initialize the CSV reader
            reader = csv.reader(f)
            # loop over the rows in the index
            for row in reader:
                features = [float(x) for x in row[1:]]
                d = self.chi2_distance(features, queryFeatures)
                results[row[0]] = d
            # close the reader
            f.close()

        results = sorted([(v, k) for (k, v) in results.items()])
        # return our (limited) results
        return results[:limit]

    def chi2_distance(self, histA, histB, eps = 1e-10):
        # compute the chi-squared distance
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
            for (a, b) in zip(histA, histB)])
        # return the chi-squared distance
        return d

    def __init__(self, indexPath):
  	# store our index path
  	    self.indexPath = indexPath

    
