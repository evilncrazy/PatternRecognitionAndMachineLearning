import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import binom

def plotMLE():
   """
   Plots a binomial distribution, and another with parameters estimated from samples
   from the first distribution.
   """
   # Number of samples
   NumSamples = 10

   # Number of independent experiments
   N = 10

   # Probability of success
   mu = 0.2

   # Plot the true binomial distribution
   x = xrange(N + 1)
   plt.plot(binom.pmf(x, N, mu), label="True Distribution")

   # Estimate the success probability from samples
   muML = sum(np.random.binomial(N, mu, NumSamples)) * 1.0 / N / NumSamples

   # Plot the maximum likelihood distribution
   plt.plot(binom.pmf(x, N, muML), label="MLE")

   plt.legend()
   plt.show()

def plotMLEVsN():
   """
   Plots the relative error between the true success probability and the
   MLE success probability for different numbers of samples. As the number
   of samples increase, the MLE becomes more accurate.
   """
   # Number of independent experiments
   N = 10

   # Probability of success
   mu = 0.2

   # Vary the number of samples
   errors = []
   x = range(10, 10000, 100)
   for numSamples in x:
      # Estimate mu by maximum likelihood from samples
      muML = sum(np.random.binomial(N, mu, numSamples)) * 1.0 / N / numSamples

      # Calculate the relative error
      errors.append(abs(muML - mu) / mu)

   # Plot the relative error
   plt.plot(x, errors, label="MLE".format(numSamples))
   plt.legend()
   plt.show()

if __name__ == "__main__":
   plotMLE()
   plotMLEVsN()
