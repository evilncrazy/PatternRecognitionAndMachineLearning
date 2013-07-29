import numpy as np
import matplotlib.pyplot as plt

def plotMeanVsN():
   """
   For different values of N, sample N uniformly random numbers in [0, 1) and
   plot their mean. Demonstrates the central limit theorem, which says that
   as N gets larger, the distribution of the sum (or mean) of the variables
   approach a normal distribution
   """
   NumTrials = 1000

   # Vary the sample size
   for N in [1, 2, 10]:
      # Sample N random numbers and then find their mean
      samples = [sum(np.random.random_sample(N)) / N for i in xrange(NumTrials)]
      plt.hist(samples, label="N = {0}".format(N), alpha=0.8)

   plt.legend()
   plt.show()

if __name__ == "__main__":
   plotMeanVsN()
