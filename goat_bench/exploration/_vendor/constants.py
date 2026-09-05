# Vendored from home-robot (MIT, Meta Platforms Inc.):
#   home_robot/utils/constants.py and home_robot/mapping/semantic/constants.py
# Merged here so the exploration package has no home_robot dependency.

# Sentinel depths written by valid_depth_mask() for out-of-range returns.
MIN_DEPTH_REPLACEMENT_VALUE = 10000
MAX_DEPTH_REPLACEMENT_VALUE = 10001


class MapConstants:
    """Channel layout of the 2D map tensor.

    Only the first NON_SEM_CHANNELS matter for coverage exploration; the
    semantic channels that follow are kept (with num_sem_categories=1) purely so
    the vendored map module runs unmodified.
    """

    NON_SEM_CHANNELS = 6  # Number of non-semantic channels at the start of maps
    OBSTACLE_MAP = 0
    EXPLORED_MAP = 1
    CURRENT_LOCATION = 2
    VISITED_MAP = 3
    BEEN_CLOSE_MAP = 4
    BLACKLISTED_TARGETS_MAP = 5
