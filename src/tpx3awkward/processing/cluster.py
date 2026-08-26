import numba
import numpy as np
import pandas as pd

TIMESTAMP_VALUE = 1.5625 * 1e-9  # each raw timestamp is 1.5625 nanoseconds
MICROSECOND = 1e-6


def _cluster(df, tw, radius, estimate_energy: bool = False, correct_timewalk: bool = False):
    cols = ["t", "x", "y", "ToT", "t"]
    # x, y, ToT, t, energy, t_corr
    col_indices = [0, 1, 2, 3, 0, 0]

    if estimate_energy:
        cols.append("e")
        col_indices[4] = len(cols) - 2
    if correct_timewalk:
        cols.append("t_corr")
        col_indices[5] = len(cols) - 2

    tw_ts_ticks = int(tw * MICROSECOND / TIMESTAMP_VALUE)

    events = df[cols].to_numpy(dtype=np.float64)
    events[:, 0] = np.floor_divide(events[:, 0], tw_ts_ticks)  # Bin timestamps into time windows

    labels = _get_cluster_labels(events, tw_ts_ticks, radius)

    return labels, np.array(col_indices), events[:, 1:]


@numba.jit(nopython=True, cache=True)
def _get_cluster_labels(events, tw, radius):
    n = len(events)
    labels = np.full(n, -1, dtype=np.int64)
    cluster_id = 0

    max_time = radius * tw  # maximum time difference allowed for clustering
    radius_sq = radius**2

    for i in range(n):
        if labels[i] == -1:  # if event is unclustered
            labels[i] = cluster_id
            for j in range(i + 1, n):  # scan forward only
                if events[j, 4] - events[i, 4] > max_time:  # early exit based on time
                    break
                # Compute squared Euclidean distance
                dx = events[i, 0] - events[j, 0]
                dy = events[i, 1] - events[j, 1]
                dt = events[i, 2] - events[j, 2]
                distance_sq = dx * dx + dy * dy + dt * dt

                if distance_sq <= radius_sq:
                    labels[j] = cluster_id
            cluster_id += 1

    return labels


@numba.jit(nopython=True, cache=True)
def _group_indices(labels):
    """
    Group indices by cluster ID using pre-allocated arrays in a Numba-optimized way.

    Parameters
    ----------
    labels : np.ndarray
        Array of cluster labels for each event.
    num_clusters : int
        Number of unique clusters.
    max_cluster_size : int
        Maximum number of events in a single cluster.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of shape (num_clusters, max_cluster_size), where each row corresponds to a cluster
        and contains event indices padded with -1 for unused slots.
    """
    num_clusters = np.max(labels) + 1  # Assume no noise, all labels are valid clusters
    max_cluster_size = np.bincount(labels).max()
    cluster_array = -1 * np.ones((num_clusters, max_cluster_size), dtype=np.int32)
    cluster_counts = np.zeros(num_clusters, dtype=np.int32)

    for idx in range(labels.shape[0]):
        cluster_idx = labels[idx]  # Label is directly the cluster ID
        cluster_array[cluster_idx, cluster_counts[cluster_idx]] = idx
        cluster_counts[cluster_idx] += 1

    return cluster_array


@numba.jit(nopython=True, cache=True)
def _centroid_clusters(
    cluster_arr: np.ndarray,
    events: np.ndarray,
    col_indices: np.array,
    estimate_energy: bool = False,
    correct_timewalk: bool = False,
) -> tuple[np.ndarray]:
    num_clusters = cluster_arr.shape[0]
    max_cluster = cluster_arr.shape[1]
    t = np.zeros(num_clusters, dtype="uint64")
    xc = np.zeros(num_clusters, dtype="float32")
    yc = np.zeros(num_clusters, dtype="float32")
    ToT_max = np.zeros(num_clusters, dtype="uint32")
    ToT_sum = np.zeros(num_clusters, dtype="uint32")
    n = np.zeros(num_clusters, dtype="ubyte")

    # must always define arrays becuase numba wants identical return signatures
    e_sum = np.zeros(num_clusters, dtype="float32") if estimate_energy else np.empty(0, dtype="float32")
    t_corr = np.zeros(num_clusters, dtype="uint64") if correct_timewalk else np.empty(0, dtype="uint64")

    for cluster_id in range(num_clusters):
        _ToT_max = np.ushort(0)
        for event_num in range(max_cluster):
            event = cluster_arr[cluster_id, event_num]
            if event > -1:  # if we have an event here
                if events[event, 2] > _ToT_max:  # find the max ToT, assign, use that time
                    _ToT_max = events[event, 2]
                    t[cluster_id] = events[event, 3]
                    if correct_timewalk:
                        t_corr[cluster_id] = events[event, col_indices[5]]
                    ToT_max[cluster_id] = _ToT_max
                xc[cluster_id] += events[event, 0] * events[event, 2]  # x and y centroids by time over threshold
                yc[cluster_id] += events[event, 1] * events[event, 2]
                ToT_sum[cluster_id] += events[event, 2]  # calcuate sum
                n[cluster_id] += np.ubyte(1)  # number of events in cluster

                if estimate_energy:
                    e_sum[cluster_id] += events[event, col_indices[4]]
            else:
                break

        if ToT_sum[cluster_id] != 0:
            xc[cluster_id] /= ToT_sum[cluster_id]  # normalize
            yc[cluster_id] /= ToT_sum[cluster_id]

    return t, xc, yc, ToT_max, ToT_sum, n, e_sum, t_corr


def _ingest_cent_data(
    data: tuple[np.ndarray], estimate_energy: bool = False, correct_timewalk: bool = False
) -> dict[str, np.ndarray]:
    """
    Package np.ndarray into a dict with keys associated with the names of the columns dataframe.

    Parameters
    ----------
    data : np.ndarray
        The stream of cluster data from cluster_arr_to_cent()
    estimate_energy : bool, optional
        Whether the data includes estimated energy sum (e_sum). Default is False.

    Returns
    -------
    Dict[str, np.ndarray]
        A dictionary with keys:
        ['t', 'xc', 'yc', 'ToT_max', 'ToT_sum', 'n']
        and optional keys 'e_sum' and 't_corr'
    """
    # first 6 columns are always included
    key_string = "t,xc,yc,ToT_max,ToT_sum,n"
    rdict = dict(zip(key_string.split(","), data[:6], strict=True))

    if estimate_energy:
        rdict["e_sum"] = data[6]
    if correct_timewalk:
        rdict["t_corr"] = data[7]

    return rdict


def cluster_decoded_df(
    df: pd.DataFrame,
    tw: float,
    radius: int,
) -> pd.DataFrame:
    """
    Cluster and centroid a decoded DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Decoded dataframe with columns ['x', 'y', 'ToT', 't', 'chip'].
        May also include optional columns 'e' and 't_corr'.
    tw : float
        Time window for the clustering algorithm, in microseconds.
    radius : int,
        Radius for the clustering algorithm, in pixels.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['t', 'xc', 'yc', 'ToT_max', 'ToT_sum', 'n'].
        Includes columns 'e_sum' and/or 't_corr' depending on the input DataFrame.
    """
    estimate_energy: bool = "e" in df.columns
    correct_timewalk: bool = "t_corr" in df.columns

    cluster_labels, col_indices, events = _cluster(
        df, tw, radius, estimate_energy=estimate_energy, correct_timewalk=correct_timewalk
    )
    df["cluster_id"] = cluster_labels
    cluster_array = _group_indices(cluster_labels)

    data = _centroid_clusters(
        cluster_array,
        events,
        col_indices,
        estimate_energy=estimate_energy,
        correct_timewalk=correct_timewalk,
    )

    return (
        pd.DataFrame(_ingest_cent_data(data, estimate_energy=estimate_energy, correct_timewalk=correct_timewalk))
        .sort_values((["t_corr", "t"] if correct_timewalk else ["t"]) + ["xc", "yc", "ToT_max", "ToT_sum"])
        .reset_index(drop=True)
    )
