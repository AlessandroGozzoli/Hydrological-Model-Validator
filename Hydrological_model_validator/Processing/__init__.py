# __init__.py

from .time_utils import (
    leapyear,
    true_time_series_length,
    split_to_yearly,
    split_to_monthly,
    get_common_years,
    get_season_mask,
)

from .data_alignment import (
    get_valid_mask,
    get_valid_mask_pandas,
    align_pandas_series,
    align_numpy_arrays,
    get_common_series_by_year,
    get_common_series_by_year_month,
    extract_mod_sat_keys,
    gather_monthly_data_across_years,
)

from .file_io import (
    mask_reader,
    load_dataset,
)

from .stats_math_utils import (
    fit_huber,
    fit_lowess,
    round_up_to_nearest,
)

from .utils import (
    find_key
)

from .Efficiency_metrics import (
    r_squared,
    weighted_r_squared,
    nse,
    index_of_agreement,
    ln_nse,
    nse_j,
    index_of_agreement_j,
    relative_nse,
    relative_index_of_agreement,
    monthly_r_squared,
    monthly_weighted_r_squared,
    monthly_nse,
    monthly_index_of_agreement,
    monthly_ln_nse,
    monthly_nse_j,
    monthly_index_of_agreement_j,
    monthly_relative_nse,
    monthly_relative_index_of_agreement,
)

from .Target_computations import (
    compute_single_target_stat,
    compute_single_month_target_stat,
    compute_normalised_target_stats,
    compute_normalised_target_stats_by_month,
    compute_target_extent_monthly,
    compute_target_extent_yearly,
)

from .Taylor_computations import (
    compute_taylor_stat_tuple,
    compute_std_reference,
    compute_norm_taylor_stats,
    build_all_points,
    compute_yearly_taylor_stats,
)