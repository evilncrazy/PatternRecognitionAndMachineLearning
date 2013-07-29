import numpy as np
import matplotlib.pyplot as plt

def plotScatterDiagonal():
   """
   Plot scatter plots of several bivariate normal distributions, each with a
   different covariance matrix with a zero off diagonal, i.e. each covariance
   matrix has the form [a 0; 0 b]
   """
   # Set the mean to zero for both variables
   mean = [0, 0]

   # Randomly sample from a bivariate normal distribution
   for a, b, c in [(100, 100, 'red'), (1, 100, 'blue'), (100, 1, 'green')]:
      # Create the covariance matrix
      cov = [[a, 0], [0, b]]

      # Randomly sample 1000 (x, y) points
      x, y = np.random.multivariate_normal(mean, cov, 1000).T

      # Plot the points
      plt.scatter(x, y, label="A = {0}, B = {1}".format(a, b), color=c)
   
   # Set the axis to be equal to prevent skewing
   plt.axis('equal')
   plt.legend()
   plt.show()

if __name__ == "__main__":
   plotScatterDiagonal()
