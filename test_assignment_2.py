# Question 1 
import numpy as np

x = np.array([3.6, 3.8, 3.9])
val = np.array([1.675, 1.436, 1.318])
w = 3.7

neville = np.zeros((len(x), len(x)))

for i in range(len(x)):
    neville[i][0] = val[i]

for i in range(1, len(x)):
    for j in range(1, i + 1):
        term1 = (w - x[i - j]) * neville[i][j - 1]
        term2 = (w - x[i]) * neville[i - 1][j - 1]
        neville[i][j] = (term1 - term2) / (x[i] - x[i - j])

final_value = neville[-1][-1]
print(f"{final_value:.16f}\n")

# Question 2
import numpy as np

def newton_forward(xi, fxi):
    lim = len(xi)
    diffs = np.zeros((lim, lim))

    for i in range(lim):
        diffs[i][0] = fxi[i]
    
    for i in range(1, lim):
        for j in range(1, i + 1):
            diffs[i][j] = (diffs[i][j-1] - diffs[i-1][j-1]) / (xi[i] - xi[i-j])


    
    for i in range(lim-3, lim):
        print(f"{diffs[i][i]:.7f}")


xi = [7.2, 7.4, 7.5, 7.6]
fxi = [23.5492, 25.3913, 26.8224, 27.4589]


newton_forward(xi, fxi)

# Question 3
answer3 = (23.5492 + 9.2105000*(7.3-7.2))
print("\n")
print(answer3)

#Question 5
import numpy as np

def cubic_spline(x, f):
    n = len(x)
    h = np.diff(x)  
    alpha = np.zeros(n-1)
    
  
    for i in range(1, n-1):
        alpha[i] = (3/h[i] * (f[i+1] - f[i]) - 3/h[i-1] * (f[i] - f[i-1]))
    
  
    A = np.zeros((n, n))
    b = np.zeros(n)
    

    A[0, 0] = 1
    A[-1, -1] = 1
    b[0] = 0
    b[-1] = 0
    

    for i in range(1, n-1):
        A[i, i-1] = h[i-1]
        A[i, i] = 2 * (h[i-1] + h[i])
        A[i, i+1] = h[i]
        b[i] = alpha[i]
    

    c = np.linalg.solve(A, b)
    

    a = f[:-1]
    b_coeff = np.zeros(n-1)
    d = np.zeros(n-1)
    
    for i in range(n-1):
        b_coeff[i] = (f[i+1] - f[i]) / h[i] - h[i] * (c[i+1] + 2 * c[i]) / 3
        d[i] = (c[i+1] - c[i]) / (3 * h[i])
    
    return A, b, c, a, b_coeff, d


x = np.array([2, 5, 8, 10])
f = np.array([3, 5, 7, 9])


A, b, c, a, b_coeff, d = cubic_spline(x, f)

print("\n")
print("Matrix A:")
print(A)
print("\nVector b:")
print(b)
print("\nVector x")
print(c)



