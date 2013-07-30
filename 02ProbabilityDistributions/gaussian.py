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

def plotScatterEigen():
   """
   Plots a scatter plot of a bivariate Gaussian distribution, along with the
   eigenvectors and eigenvalues of the covariance matrix, which represent the
   axis of the elliptical contour when the probability density is equal to
   exp(-0.5)
   """
   # Set the mean to zero
   mean = [0, 0]

   # Use a diagonal covariance
   cov = [[100, 50], [30, 100]]

   # Plot a scatter of the distribution
   x, y = np.random.multivariate_normal(mean, cov, 1000).T

   # Plot the points
   plt.scatter(x, y, alpha=0.2)

   # Find the eigenvectors and eigenvalues of the covariance matrix
   w, v = np.linalg.eig(cov)
   x0, y0 = v.item(0, 0) * np.sqrt(w[0]), v.item(1, 0) * np.sqrt(w[0])
   x1, y1 = v.item(0, 1) * np.sqrt(w[1]), v.item(1, 1) * np.sqrt(w[1])

   # Plot the eigenvectors and eigenvalues
   plt.plot([-x0, x0], [-y0, y0], linewidth=3)
   plt.plot([-x1, x1], [-y1, y1], linewidth=3)

   plt.show()

if __name__ == "__main__":
   plotScatterEigen()
   plotScatterDiagonal()
