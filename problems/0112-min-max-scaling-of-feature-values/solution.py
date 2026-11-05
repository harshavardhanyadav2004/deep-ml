def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A list of numerical values
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    result = []
    minimum , maximum = min(x) , max(x)
    for element in x :
        result.append((element-minimum)/(maximum-minimum))
    return result