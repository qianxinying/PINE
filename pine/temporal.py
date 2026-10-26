"""Dependency-light query-relative temporal utilities."""



def relative_intervals(query_time, edge_time, time_range=None):
    """Compute the signed matrix ``t_q - t_epsilon``.

    Parameters
    ----------
    query_time:
        One timestamp per query, with any shape that can be flattened.
    edge_time:
        One timestamp per directed support edge.
    time_range:
        Optional maximum permitted absolute interval.
    """
    query_time = query_time.long().reshape(-1, 1)
    edge_time = edge_time.long().reshape(1, -1)
    delta = query_time - edge_time
    if time_range is not None and delta.numel():
        max_abs_delta = int(delta.abs().max().item())
        if max_abs_delta > int(time_range):
            raise ValueError(
                "query-relative interval exceeds query_time_range: "
                f"max |t_q-t_e|={max_abs_delta}, "
                f"query_time_range={int(time_range)}"
            )
    return delta
