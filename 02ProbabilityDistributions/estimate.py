import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

def estimateMLE(data):
   """
   Return the maximum likelihood estimate of the mean
   """
   return sum(data) / len(data)

def estimateBayes(data, mu0, var0, var):
   """
   Estimate the mean from a Bayesian approach, using a Gaussian prior
   """
   N = len(data)

   # Use the maximum likelihood estimate of the mean in the likelihood
   muML = estimateMLE(data)

   # Calculate the mean and variance
   muN = var / (N * var0 + var) * mu0 + (N * var0) / (N * var0 + var) * muML
   varN = 1.0 / (1.0 / var0 + N / var)

   return muN, varN

def plotEstimateVsN():
   """
   Estimate the mean of a Gaussian distribution, plotting the resulting distributions
   as the number of data points increase. The true variance is known in advance.
   """
   # Generate points from a Gaussian distribution
   mu = -0.8 # Mean
   var = 0.1 # Variance

   # Use mean of 0 for prior, and the true variance for both prior and likelihoods
   data = np.random.normal(mu, var, 10)

   # Plot the prior distribution with mean 0 and true variance
   x = np.linspace(-1, 1, 100)
   plt.plot(x, norm.pdf(x, 0, np.sqrt(var)), label="N = 0")
   
   # Plot distribution as N gets larger
   for i in [1, 2, 10]:
      # Estimate the mean and variance from i data points
      mu, v = estimateBayes(data[:i], 0, var, var)

      # Plot the normal distribution curve
      plt.plot(x, norm.pdf(x, mu, np.sqrt(v)), label="N = {0}".format(i))

   plt.legend()
   plt.show()

if __name__ == "__main__":
   plotEstimateVsN()
