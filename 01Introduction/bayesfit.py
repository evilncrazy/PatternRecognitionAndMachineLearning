import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
   # Target function to approximate
   targetFunc = lambda x: np.sin(2 * np.pi * x)

   # Number of training data points
   N = 10

   # Order of the polynomial
   M = 8

   # Standard deviation of training data points (controls amount of noise)
   Std = 0.1

   # Generate training data with Gaussian noise
   xdata = np.random.random_sample(N)
   ydata = targetFunc(xdata) * np.random.normal(1, Std, N)
   points = zip(xdata, ydata)

   # Hyperparameters
   alpha = 5e-3
   beta = 11.1

   # Evenly spaced x values
   xs = np.linspace(0, 1, 10)

   # Stores y vertices of the region within one standard deviation of the
   # mean of the predictive distribution. We can reconstruct the region polygon
   # by piecing together the top and bottom halves
   regionTop = []
   regionBot = []

   # Stores the y values of the mean of the predictive distribution for each x
   ymean = []

   phi = lambda x: np.matrix(list(x ** i for i in xrange(M + 1))).T
   for nx in xs:
      S = (alpha * np.eye(M + 1) + beta * sum(phi(x) for x, t in points) * phi(nx).T).I
      m = (beta * phi(nx).T * S * sum(phi(x) * t for x, t in points)).item(0, 0)
      s = np.sqrt((1.0 / beta + phi(nx).T * S * phi(nx)).item(0, 0))

      ymean.append(m)

      # Find the region within one standard deviation of the mean
      regionTop.append(m + s)
      regionBot.append(m - s)

   # Plot actual function
   plt.plot(xs, targetFunc(xs), label="Target Function")

   # Plot the sample data points
   plt.scatter(xdata, ydata, label="Observations")

   # Plot the mean
   plt.plot(xs, ymean, label="Mean")

   # Plot the uncertainty region 
   plt.fill(list(xs) + list(xs[::-1]), regionTop + regionBot[::-1], alpha=.5)

   plt.legend()
   plt.show()
