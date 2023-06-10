import numpy as np


def empca_w(a, ncomps=None, w = None, emtol=1e-6, maxiters=100, Weighted = False):
    # Reshape input matrix to 2D
    a = a.reshape(a.shape[0], -1)

    # If ncomps is not specified, use the minimum dimension of the input matrix
    if ncomps is None:
        ncomps = min(a.shape)

    # Initialize matrices to store the principal components (u) and their coefficients (sv)
    u = np.zeros((a.shape[0], ncomps))
    sv = np.zeros((a.shape[1], ncomps))
    
    if ~Weighted:
        # Create a boolean mask for missing values (NaN)
        w = ~np.isnan(a)
        # Replace missing values with zero
        a[~w] = 0
    else:
        # Create a boolean mask for missing values (NaN)
        b = w > 0
        # Replace missing values with zero
        a[~b] = 0


    # Loop over each component
    for comp in range(ncomps):
        # Initialize the component vector with random values and normalize
        u[:, comp] = np.random.randn(a.shape[0])
        u[:, comp] /= np.linalg.norm(u[:, comp])

        # Loop until convergence or maximum number of iterations
        for _ in range(maxiters):
            # Copy of the current component vector for convergence check
            u0 = np.copy(u[:, comp])

            # E-step
            sv[:, comp] = a.T @ u[:, comp]

            # Weighting sv for NaN handling
            svw = np.multiply(sv[:, comp].reshape(-1, 1), w.T)
            # M-step
            u[:, comp] = np.sum(a * svw.T, axis=1) / (svw.T @ sv[:, comp])
            # Normalize the updated component vector
            u[:, comp] /= np.linalg.norm(u[:, comp])

            # Check for convergence: if the maximum change in the component vector is <= emtol, break the loop
            if np.max(np.abs(u0 - u[:, comp])) <= emtol:
                break

        # Print information about the convergence
        print(f"eigenvector {comp + 1} kept after {_ + 1} iterations")

        # Remove the contribution of the current component from the data
        a = a - np.outer(u[:, comp], sv[:, comp])

        if ~Weighted:
            # Set the missing values back to zero
            a[~w] = 0
        else:
            # Set the missing values back to zero
            a[~b] = 0

    if ~Weighted:
        # Restore the missing values in the data
        a[~w] = np.nan
    else:
        # Restore the missing values in the data
        a[~b] = np.nan
    # Compute the singular values and right singular vectors from sv
    s = np.diag(np.sqrt(np.sum(sv**2, axis=0)))
    v = sv / np.sqrt(np.sum(sv**2, axis=0))

    # Return the left singular vectors (u), singular values (s), right singular vectors (v), and the residual data (a)
    return u, s, v, a

