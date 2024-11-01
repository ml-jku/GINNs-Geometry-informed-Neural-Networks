def format_significant_digits(number, n_digits):
    if number == 0:
        return "0.0"

    # Find the exponent to adjust the precision
    exponent = int(f"{number:e}".split('e')[-1])

    # Format the number to show the first non-zero digit followed by (n_digits - 1) additional digits
    n_decimals = max(n_digits - exponent - 1, n_digits)
    formatted_number = f"{number:.{n_decimals}f}"

    # Return the formatted number
    return formatted_number