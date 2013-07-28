import numpy as np
import matplotlib.pyplot as plt

targetFunc = lambda x: np.sin(2 * np.pi * x)

def polyfitSumSquares(M, points):
   """
   Given a list of (x, t) pairs of data points sampled from a function,
   returns a vector of coefficients of an order M polynomial that approximates
   the function by minimizing the sum of squares error
   """
   A = np.zeros((M + 1, M + 1))
   b = np.zeros(M + 1)
   for i in xrange(M + 1):
      for j in xrange(M + 1):
         # A(i, j) and b(i) are given in Exercise 1.1
         A[i][j] = sum(pow(x, i + j) for x, t in points)
      b[i] = sum(pow(x, i) * t for x, t in points)

   # Solve for the polynomial coefficients
   return np.linalg.solve(A, b)

def polyfitSumSquaresReg(M, points, lam):
   """
   Similar to sum squares, but adds a regularisation term to prevent overfitting.
   The first polynomial coefficient term is not regularised.
   lam controls the impact of regularisation
   """
   A = np.zeros((M + 1, M + 1))
   b = np.zeros(M + 1)
   for i in xrange(M + 1):
      for j in xrange(M + 1):
         A[i][j] = sum(pow(x, i + j) for x, t in points)
         if i > 0:
            # Add regularisation term (Exercise 1.2)
            A[i][j] += 2 * lam
      b[i] = sum(pow(x, i) * t for x, t in points)

   # Solve for the polynomial coefficients
   return np.linalg.solve(A, b)

def rmsError(poly, x, y):
   """
   Calculate the Root Mean Squared error for (poly(x), y) points
   """
   return np.sqrt(sum(pow(poly(x) - y, 2)) / len(x))

def generateTest(N):
   """
   Generate N test points from N uniformly sampled points between [0, 1)
   """
   # Calculate y value for each x value
   xdata = np.random.random_sample((N,))
   ydata = targetFunc(xdata)
   return xdata, ydata

def generateTrain(N, std):
   """
   Generate N training points from N uniformly sampled points between [0, 1),
   with Gaussian noise applied to each of the y values (with standard deviation std)
   """
   xdata, ydata = generateTest(N)
   return xdata, ydata * np.random.normal(1, std, N)

def trainPolyfit(M, points):
   """
   Returns the polynomial that minimize the sum of squares error
   """
   return np.poly1d(polyfitSumSquares(M, points)[::-1])

def trainPolyfitReg(M, points, lam):
   """
   Returns the polynomial that minimize the sum of squares error with regularisation
   term lam
   """
   return np.poly1d(polyfitSumSquaresReg(M, points, lam)[::-1])

def plotNoRegVsReg(N, M):
   """
   Compare both unregularised and regularised versions of the algorithms against
   the true target function.
   """
   # Generate the training data 
   xdata, ydata = generateTrain(N, 0.1)
   points = zip(xdata, ydata)

   pss = trainPolyfit(M, points)
   pssreg = trainPolyfitReg(M, points, 1)

   # Plot these polynomials against the actual function and the sampled points
   x = np.linspace(0, 1, 100) # Evenly spaced x coordinates

   plt.scatter(xdata, ydata, label="Observations")
   plt.plot(x, targetFunc(x), label="Target Function") 
   plt.plot(x, pss(x), label="Polyfit")
   plt.plot(x, pssreg(x), label="Regularised Polyfit")

   plt.legend()
   plt.show()

def plotNoRegVsM(N):
   """
   Plots the unregularised polynomial for different values of M, demonstrating
   that for higher M, there is a tendency to overfit
   """
   MVals = [0, 1, 3, 9]

   # Generate the training data 
   xdata, ydata = generateTrain(N, 0.1)
   points = zip(xdata, ydata)

   polys = []
   for M in MVals:
      polys.append(trainPolyfit(M, points))

   # Plot these polynomials against the actual function and the sampled points
   x = np.linspace(0, 1, 100) # Evenly spaced x coordinates

   plt.scatter(xdata, ydata, label="Observations")
   plt.plot(x, targetFunc(x), label="Target Function")
   
   # Plot each of the polynomials
   for poly, M in zip(polys, MVals):
      plt.plot(x, poly(x), label="M = {0}".format(M))

   plt.ylim((-1.5, 1.5))
   plt.legend()
   plt.show()

def plotRmsVsM(N):
   """
   Plot the RMS error value for different values of M, showing error for both
   training and test data
   """
   MVals = xrange(10)

   # Generate the training data 
   xdata, ydata = generateTrain(N, 0.1)
   points = zip(xdata, ydata)

   # Generate the test data
   xtest, ytest = generateTest(N)
   
   errTrain, errTest = [], []
   for M in MVals:
      # Train the unregularised sum of squares polynomial
      pss = trainPolyfit(M, points)

      errTrain.append(rmsError(pss, xdata, ydata))
      errTest.append(rmsError(pss, xtest, ytest))

   # Plot RMS for various values of M
   plt.plot(MVals, errTrain, label="Training")
   plt.plot(MVals, errTest, label="Test")
   plt.ylim((0, 1))
   plt.legend()
   plt.show()

def plotRmsVsLambda(N):
   """
   Plot the RMS error value for different values of lambda, showing error for both
   training and test data
   """
   M = 9

   # Let the x-axis be the log of the lambda values
   logLamVals = xrange(-35, 0)

   # Generate the training data 
   xdata, ydata = generateTrain(N, 0.1)
   points = zip(xdata, ydata)

   # Generate the test data
   xtest, ytest = generateTest(N)

   errTrain, errTest = [], []
   for lam in logLamVals:
      # Train the regularised sum of squares polynomial
      pssReg = trainPolyfitReg(M, points, np.exp(lam))

      errTrain.append(rmsError(pssReg, xdata, ydata))
      errTest.append(rmsError(pssReg, xtest, ytest))

   # Plot RMS for various values of lamdba
   plt.plot(logLamVals, errTrain, label="Training")
   plt.plot(logLamVals, errTest, label="Test")
   plt.ylim((0,1))
   plt.legend()
   plt.show()

if __name__ == "__main__":
   plotNoRegVsM(10)
   plotNoRegVsReg(10, 3)
   plotRmsVsM(20)
   plotRmsVsLambda(10)
