def payload_bits(payload: dict[str, object], quantized_bit_width: int) -> float:
    # Compression is measured in BITS, not element counts, and the
    # difference is not academic: a quantizer stores the same number of
    # values at a quarter of the width, so counting elements reports it
    # as achieving nothing -- or, once per-channel scales and residuals
    # are added, as making the cache LARGER. An earlier version of the
    # combination pipeline counted elements and scored KIVI at 0.66x,
    # i.e. as inflation, which is what surfaced this.
    #
    # The accounting exploits a real property of these payloads rather
    # than special-casing each compressor: quantized values are stored as
    # ints, while everything kept at full precision (scales, zero-points,
    # residual rows, low-rank factors) is stored as floats. So ints are
    # charged the quantized width and floats the full 32, which is
    # faithful for a quantizer, for a low-rank method whose payload is
    # entirely float, and for a hybrid carrying both.
    if quantized_bit_width < 1:
        raise ValueError("quantized_bit_width must be at least 1")

    def bits(value: object) -> float:
        if isinstance(value, list):
            return sum(bits(item) for item in value)
        if isinstance(value, bool):
            return 0.0
        if isinstance(value, int):
            return float(quantized_bit_width)
        if isinstance(value, float):
            return 32.0
        return 0.0

    return sum(bits(value) for value in payload.values())


def dense_bits(rows: int, channels: int, bit_width: int = 32) -> float:
    # What the same tensor would cost stored densely at full precision --
    # the denominator every compression ratio here is taken against.
    if rows < 0 or channels < 0:
        raise ValueError("rows and channels must be non-negative")
    if bit_width < 1:
        raise ValueError("bit_width must be at least 1")
    return float(rows * channels * bit_width)
