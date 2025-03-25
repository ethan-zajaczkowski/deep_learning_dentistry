from deep_learning_dentistry.data_curation.data_processing.utils.data_functions import tooth_quadrant_determiner


def tooth_value_mapper(values, tooth_id, variable=None):
    """
    Maps raw tooth values to processed values based on variable type.
    """
    quadrant = tooth_quadrant_determiner(tooth_id)

    if variable == "pockets and recession":
        return process_pockets_and_recession(values, quadrant)
    elif variable == "bleeding and suppuration":
        return process_bleeding_and_suppuration(values, quadrant)


def process_bleeding_and_suppuration(values, quadrant):
    """
    Processes values for bleeding and suppuration measurements.
    Simplified using a mapping table and index calculation.
    """
    # Define mapping: (quadrant, surface) -> ordered site codes
    mapping = {
        # Format: (quadrant, surface): [site_codes_for_remainder_1, 2, 0]
        ("Q1", "Buccal"): ["DB", "B", "MB"],
        ("Q1", "Palatal"): ["DP", "P", "MP"],
        ("Q2", "Buccal"): ["MB", "B", "DB"],
        ("Q2", "Palatal"): ["MP", "P", "DP"],
        ("Q3", "Buccal"): ["MB", "B", "DB"],
        ("Q3", "Lingual"): ["ML", "L", "DL"],
        ("Q4", "Buccal"): ["DB", "B", "MB"],
        ("Q4", "Lingual"): ["DL", "L", "ML"],
    }

    remainder = values[1] % 3
    surface = values[0]

    # Get site code order or None if invalid combination
    site_order = mapping.get((quadrant, surface), None)

    if site_order:
        # Calculate index using remainder logic
        return site_order[(remainder - 1) % 3]
    return None  # or raise ValueError for invalid input


def process_pockets_and_recession(values, quadrant):
    """
    Processes values for pockets and recession measurements.
    """
    if quadrant in ["Q1", "Q4"]:
        ordered = [values[0], values[1], values[2]]
    elif quadrant in ["Q2", "Q3"]:
        ordered = [values[2], values[1], values[0]]
    else:
        ordered = values.copy()  # Fallback for unexpected quadrants
    return ordered
