import pytest
import numpy as np
from datetime import datetime, timedelta

from Hydrological_model_validator.Processing.Missing_data import check_missing_days, find_missing_observations, eliminate_empty_fields

###############################################################################
# Tests for check_missing_days
###############################################################################


# Test check_missing_days correctly aligns timestamps when series is complete and start date offset
def test_check_missing_days_complete_series():
    base_datetime = datetime(2000, 1, 1)
    
    # Simulate original timestamps offset by 1 hour before desired start date
    base_time = (base_datetime - timedelta(hours=1)).timestamp()
    
    # Create timestamps spaced by exactly one day, 5 points total
    T_orig = np.array([base_time + i*86400 for i in range(5)])
    
    # Generate dummy data for each timestamp
    data_orig = np.random.rand(5, 2, 2)
    
    # Run check_missing_days to realign timestamps starting exactly at base_datetime
    Ttrue, data_complete = check_missing_days(T_orig, data_orig, desired_start_date=base_datetime)
    
    # Confirm first timestamp was shifted correctly to desired start date
    assert datetime.utcfromtimestamp(Ttrue[0]) == base_datetime
    
    # Confirm no days were added or removed (lengths unchanged)
    assert len(Ttrue) == len(T_orig)

# Test check_missing_days inserts NaN data for missing days in time series
def test_check_missing_days_with_gaps():
    base_time = datetime(2000, 1, 1).timestamp()
    
    # Timestamps have a gap between day 2 and 3 (missing day 2)
    T_orig = np.array([base_time, base_time + 86400, base_time + 3*86400, base_time + 4*86400])
    
    # Generate dummy data for each timestamp
    data_orig = np.random.rand(4, 2, 2)
    
    # Fill missing days with NaNs to maintain continuity
    Ttrue, data_complete = check_missing_days(T_orig, data_orig)
    
    # Confirm total timestamps now cover 5 days, including the missing day
    assert len(Ttrue) == 5
    
    # Confirm that the inserted missing day's data is all NaNs
    assert np.isnan(data_complete[2]).all()

# Test check_missing_days shifts timestamps to start exactly at desired start date
def test_check_missing_days_shift_start_date():
    base_time = datetime(1999, 12, 30).timestamp()
    
    # Original timestamps start before desired date by two days
    T_orig = np.array([base_time + i*86400 for i in range(3)])
    
    # Dummy data corresponding to timestamps
    data_orig = np.random.rand(3, 2, 2)
    
    desired_start = datetime(2000, 1, 1)
    
    # Shift timestamps forward to align with desired start date
    Ttrue, _ = check_missing_days(T_orig, data_orig, desired_start_date=desired_start)
    
    # Verify first timestamp exactly matches desired start date
    assert datetime.utcfromtimestamp(Ttrue[0]) == desired_start

# Test check_missing_days raises ValueError if timestamps are not strictly increasing
def test_check_missing_days_non_increasing_timestamps():
    base_time = datetime(2000, 1, 1).timestamp()
    
    # Timestamps with repeated value (non-strictly increasing)
    T_orig = np.array([base_time, base_time, base_time + 86400])
    
    data_orig = np.random.rand(3, 2, 2)
    
    # Should raise because timestamps must strictly increase
    with pytest.raises(ValueError):
        check_missing_days(T_orig, data_orig)

# Test check_missing_days raises TypeError for invalid input types
def test_check_missing_days_wrong_input_types():
    # Passing list instead of numpy array for timestamps should error
    with pytest.raises(TypeError):
        check_missing_days([1,2,3], np.random.rand(3,2,2))
    
    # Passing non-numpy array for data should error
    with pytest.raises(TypeError):
        check_missing_days(np.array([1,2,3]), [[1,2],[3,4]])


###############################################################################
# Tests for find_missing_observations
###############################################################################


# Test find_missing_observations detects no missing days in fully valid data
def test_find_missing_observations_none_missing():
    data = np.ones((3, 2, 2))  # all data present, no zeros or NaNs
    cnan, satnan = find_missing_observations(data)
    
    # Expect no missing days count
    assert cnan == 0
    
    # Expect empty list of missing day indices
    assert satnan == []

# Test find_missing_observations detects missing days with all zeros or all NaNs
def test_find_missing_observations_some_missing():
    data = np.ones((4, 2, 2))
    
    # Set day 1 data to all zeros (interpreted as missing)
    data[1] = 0
    
    # Set day 3 data to all NaNs (also missing)
    data[3] = np.nan
    
    cnan, satnan = find_missing_observations(data)
    
    # Should detect exactly two missing days
    assert cnan == 2
    
    # Indices of missing days should be 1 and 3
    assert set(satnan) == {1, 3}

# Test find_missing_observations detects all days missing (all zeros)
def test_find_missing_observations_all_missing():
    data = np.zeros((2, 2, 2))  # entire dataset zeroed out
    
    cnan, satnan = find_missing_observations(data)
    
    # All days should be counted as missing
    assert cnan == 2
    
    # Missing day indices are all days: 0 and 1
    assert satnan == [0, 1]

# Test find_missing_observations raises TypeError for invalid input type
def test_find_missing_observations_wrong_type():
    # Passing list instead of np.ndarray should raise TypeError
    with pytest.raises(TypeError):
        find_missing_observations([[1,2],[3,4]])

# Test find_missing_observations raises ValueError for wrong number of dimensions
def test_find_missing_observations_wrong_dim():
    # Passing 2D array instead of expected 3D should raise ValueError
    with pytest.raises(ValueError):
        find_missing_observations(np.ones((3, 2)))


###############################################################################
# Tests for eliminate_empty_fields
###############################################################################


# Test eliminate_empty_fields leaves data unchanged when no empty days present
def test_eliminate_empty_fields_none_empty():
    data = np.random.rand(3, 2, 2)  # all days contain valid data
    result = eliminate_empty_fields(data.copy())
    
    # Result should be unchanged since no days are empty
    assert np.allclose(result, data)

# Test eliminate_empty_fields replaces all-zero or all-NaN days with NaNs
def test_eliminate_empty_fields_some_empty():
    data = np.random.rand(4, 2, 2)
    
    # Simulate empty days by setting day 0 to all zeros (missing)
    data[0] = 0
    
    # Simulate empty days by setting day 2 to all NaNs (missing)
    data[2] = np.nan
    
    result = eliminate_empty_fields(data.copy())
    
    # Days that were all zeros or NaNs should now be all NaNs explicitly
    assert np.isnan(result[0]).all()
    assert np.isnan(result[2]).all()
    
    # Days that had valid data should remain unchanged
    assert np.allclose(result[1], data[1])
    assert np.allclose(result[3], data[3])

# Test eliminate_empty_fields converts all data to NaN if all days are empty
def test_eliminate_empty_fields_all_empty():
    data = np.zeros((2, 2, 2))  # all data zeroed out (all days empty)
    result = eliminate_empty_fields(data.copy())
    
    # Entire result should be NaN, as all days are empty
    assert np.isnan(result).all()

# Test eliminate_empty_fields raises TypeError if input is not a numpy array
def test_eliminate_empty_fields_wrong_type():
    # Passing a list should raise TypeError since function expects np.ndarray
    with pytest.raises(TypeError):
        eliminate_empty_fields([[1, 2], [3, 4]])

# Test eliminate_empty_fields raises ValueError if input array has incorrect dimensions
def test_eliminate_empty_fields_wrong_dim():
    # Passing a 2D array instead of expected 3D array raises ValueError
    with pytest.raises(ValueError):
        eliminate_empty_fields(np.ones((3, 2)))
