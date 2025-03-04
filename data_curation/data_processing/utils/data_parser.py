import pandas as pd
import re

## Bleeding and Suppuration Data Processing ##

def extract_surface_type_regex(cell_area_value):
    """
    Extracts 'Buccal', 'Palatal' or 'Lingual' from the given cell value using regex.
    Returns the extracted surface type (e.g., "Buccal" or "Palatal").
    """
    if isinstance(cell_area_value, str):
        match = re.search(r"(Buccal|Palatal|Lingual)", cell_area_value)
        if match:
            return match.group(1)
    return None


def determine_site(location, surface_value):
    """
    Determines the site based on the location (Buccal, Palatal, or Lingual) and the modulo result of the surface value.
    Returns the corresponding site number.
    """
    mod_result = surface_value % 3

    if location == "Buccal":
        if mod_result == 1:
            return 1
        elif mod_result == 2:
            return 2
        elif mod_result == 0:
            return 3
    elif location in ["Palatal", "Lingual"]:
        if mod_result == 1:
            return 4
        elif mod_result == 2:
            return 5
        elif mod_result == 0:
            return 6
    else:
        return None

## Chart Data Processor ##

def map_surface_to_site_endo(surface):
    """
    Maps a given surface to a list [x, y], where:
    - x is 1 if surface is 'I' (Incisal), else 0.
    - y is 1 if surface is 'O' (Occlusal), else 0.
    Handles null or NA values gracefully.
    """
    if surface is None or pd.isna(surface):  # Check for None or pd.NA
        return [0, 0]

    x = 1 if surface == "I" else 0
    y = 1 if surface == "O" else 0
    return [x, y]

def map_surface_to_site_restore(tooth_condition, surface):
    """
    Maps a given surface to a list [x1, x2, x3, x4, x5], where:
    Handles null or NA values gracefully.
    """
    if tooth_condition is None or pd.isna(tooth_condition):  # Check for None or pd.NA
        return [0, 0, 0, 0, 0]

    x1 = 1 if tooth_condition == "Bridge Retainer 3/4 Crown" else 0
    x2 = 1 if tooth_condition == "Bridge Retainer Veneer" else 0
    x3 = 1 if tooth_condition == "Veneer" else 0
    x4 = 1 if tooth_condition == "Fillings / Caries" else 0
    x5 = 1 if tooth_condition == "Inlay" else 0
    return [x1, x2, x3, x4, x5]


## Demographics Data Processer ##

def clean_value(value):
    """
    Helper function to replace missing values with 'Missing'.
    """
    if pd.isna(value) or value == "":
        return "Missing"
    return value

    
## Mobility and MAG Tooth Data Processing ##

def process_mobility_value(cell):
    """
    """
    if cell is pd.NA:
        return 0
    elif cell == "+":
        return 0.5
    elif cell == "1":
        return 1
    elif cell == "+1":
        return 1.5
    elif cell == "2":
        return 2
    elif cell == "+2":
        return 2.5
    elif cell == "3":
        return 3
    elif cell == "1,1":
        return "Error"
    
def process_mag_value(cell):
    """
    """
    if cell is pd.NA:
        return 0
    elif cell == "X":
        return 1
    else:
        return 'Error'

## Pockets and Recession Tooth Data Processing ##

def process_tooth_integer_values(cell):
    """
    Process the cell value, and return three values corresponding to the left, middle, and right sites of a tooth.
    This function exists only to extract the value of the cell without context to the type of exam.

    Handles cases where the cell contains a single value ("1") or a two-character string ("10").

    Five possible cases:
    - Cell does not exist (indicates tooth truly DNE).
    - Cell is empty.
    - Cell has only one value (also in cases where just "1" or "10" is recorded).
    - Cell has two values.
    - Cell has all three values.
    """
    if cell is pd.NA or str(cell).strip() == '' or str(cell).strip() == 'F':
        return ["Missing"] * 3

    if len(cell) == 1 or len(cell.strip()) == 2:
        value = cell.strip()
        return ["Missing", "Missing", value]

    if len(cell) == 8:
        return process_tooth_values_eight(cell)
    elif len(cell) == 11:
        return process_tooth_values_eleven(cell)

    return ["Error"] * 3
    
def process_tooth_values_eight(cell):
    """
    Process cell value if the sum of characters is 8.
    - Returns three values corresponding to the left, middle and right sites of tooth.
    - Mainly for Pockets but also for Recession
    """
    cell_stripped = cell.strip()

    # Get the list of integers in the cell.
    integers = re.findall(r'-?\d+', cell_stripped)
    integer_count = len(integers)

    # Determine the number of leading, trailing, and in-between spaces in cells.
    matches = list(re.finditer(r'-?\d+', cell))
    leading_spaces = len(cell) - len(cell.lstrip())
    trailing_spaces = len(cell) - len(cell.rstrip())
    in_between_spaces = sum(matches[i + 1].start() - matches[i].end() for i in range(len(matches) - 1))

    if integer_count == 3:
        return [int(x) for x in integers]
    elif integer_count == 2:
        if (leading_spaces == 0) and (in_between_spaces >= 4): # Left and Right
            return [integers[0], "Missing", integers[1]]
        elif (leading_spaces == 0) and (trailing_spaces >= 3): # Left and Middle
            return [integers[0], integers[1], "Missing"]
        elif leading_spaces > 1 and (trailing_spaces <= 1): # Middle and Right
            return ["Missing", integers[0], integers[1]] 
    elif integer_count == 1:
        if (leading_spaces <= 1) and (trailing_spaces >= 3): # Left
            return [integers[0], "Missing", "Missing"]
        elif (leading_spaces > 1) and (trailing_spaces <= 1): # Right
            return ["Missing", "Missing", integers[0]]
        elif (leading_spaces > 1) and (trailing_spaces <= 4): # Middle
            return ["Missing", integers[0], "Missing"]
    else:
        return ["Error"] * 3

def process_tooth_values_eleven(cell):
    """
    Process cell value if the sum of characters is 11.
    - Returns three values corresponding to the left, middle, and right sites of a tooth.
    - Handles negative and positive integers.
    - Mainly for Recession Data
    """
    cell_stripped = cell.strip()

    # Extract integers (supports negative numbers)
    integers = re.findall(r'-?\d+', cell_stripped)
    integer_count = len(integers)

    # Determine spaces
    matches = list(re.finditer(r'-?\d+', cell))
    leading_spaces = matches[0].start() if matches else len(cell)  # Spaces before the first number
    trailing_spaces = len(cell) - matches[-1].end() if matches else len(cell)  # Spaces after the last number
    in_between_spaces = sum(matches[i + 1].start() - matches[i].end() for i in range(len(matches) - 1)) if len(matches) > 1 else 0

    # Handle cases based on the number of integers
    if integer_count == 3:
        return [int(x) for x in integers]
    elif integer_count == 2:
        if (leading_spaces <= 1) and (in_between_spaces >= 6):  # Left and Right
            return [int(integers[0]), "Missing", int(integers[1])]
        elif (leading_spaces <= 1) and (trailing_spaces >= 5):  # Left and Middle
            return [int(integers[0]), int(integers[1]), "Missing"]
        elif (leading_spaces >= 4) and (trailing_spaces <= 2): # Middle and Right
            return ["Missing", int(integers[0]), int(integers[1])]
    elif integer_count == 1:
        if (leading_spaces <= 2) and (trailing_spaces > 8):  # Left
            return [int(integers[0]), "Missing", "Missing"]
        elif (leading_spaces >= 7) and (trailing_spaces <= 2): # Right
            return ["Missing", "Missing",int(integers[0])]
        elif (leading_spaces >= 3) and (trailing_spaces <= 7): # Middle
            return ["Missing", int(integers[0]), "Missing"]
    else:
        return ["Error"] * 3
