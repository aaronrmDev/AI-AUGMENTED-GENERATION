def compression_ratio(
    original_bit_width: float,
    original_element_count: int,
    compressed_bit_width: float,
    compressed_element_count: int,
) -> float:
    if compressed_bit_width <= 0 or compressed_element_count <= 0:
        raise ValueError("compressed size must be a positive number of bits")
    original_bits = original_bit_width * original_element_count
    compressed_bits = compressed_bit_width * compressed_element_count
    return original_bits / compressed_bits


def reconstruction_error(original: list[list[float]], reconstructed: list[list[float]]) -> float:
    if not original or not reconstructed:
        raise ValueError("original and reconstructed must be non-empty")
    if len(original) != len(reconstructed):
        raise ValueError("original and reconstructed must have the same number of rows")
    total = 0.0
    count = 0
    for original_row, reconstructed_row in zip(original, reconstructed, strict=True):
        if len(original_row) != len(reconstructed_row):
            raise ValueError("original and reconstructed rows must have the same length")
        for original_value, reconstructed_value in zip(
            original_row, reconstructed_row, strict=True
        ):
            total += abs(original_value - reconstructed_value)
            count += 1
    return total / count
