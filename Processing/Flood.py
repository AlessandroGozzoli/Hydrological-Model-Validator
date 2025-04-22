import numpy as np

def flood(matrice_in, loop):
    """
    Expands data over NaN regions iteratively.
    
    Parameters:
    - matrice_in: 2D numpy array with NaN values representing missing data.
    - loop: Number of iterations to expand values over NaNs.
    
    Returns:
    - tot: 2D numpy array with interpolated values in NaN regions.
    """
    # Expand the domain by 1 point with a padding of NaNs
    dummy = np.full((matrice_in.shape[0] + 2, matrice_in.shape[1] + 2), np.nan)
    dummy[1:-1, 1:-1] = matrice_in
    
    # Find indices of NaN values
    nan_idx = np.isnan(dummy)

    # Define neighbor offsets for an 8-connected neighborhood
    M, N = dummy.shape  # Get dimensions of the expanded matrix
    neighbor_offsets = np.array([M, M+1, 1, -M+1, -M, -M-1, -1, M-1])

    for _ in range(loop):
        nan_positions = np.where(nan_idx)
        if len(nan_positions[0]) == 0:
            break

        new_values = np.zeros_like(nan_positions[0], dtype=np.float64)
        valid_neighbors = np.zeros_like(nan_positions[0], dtype=np.int32)

        # Iterate over neighbors
        for offset in neighbor_offsets:
            neighbors_x = nan_positions[0] + offset  # Row indices
            neighbors_y = nan_positions[1]  # Column indices

            # 🛑 **Boundary Check: Keep valid indices only** 🛑
            valid_mask = (neighbors_x >= 0) & (neighbors_x < M)  # Row bounds
            valid_mask &= (neighbors_y >= 0) & (neighbors_y < N)  # Column bounds

            neighbors_x = neighbors_x[valid_mask]
            neighbors_y = neighbors_y[valid_mask]

            if len(neighbors_x) > 0:
                new_values[valid_mask] += dummy[neighbors_x, neighbors_y]
                valid_neighbors[valid_mask] += 1

        # Compute mean over valid neighbors
        mean_values = np.divide(new_values, valid_neighbors, out=np.full_like(new_values, np.nan), where=valid_neighbors > 0)

        # Update dummy array
        dummy[nan_positions] = mean_values
        nan_idx = np.isnan(dummy)  # Update NaN positions for next loop

    return dummy[1:-1, 1:-1]  # Remove padding