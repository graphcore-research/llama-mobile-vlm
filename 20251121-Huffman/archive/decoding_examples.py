"""
Vibe-coded examples for Huffman, tANS, and Arithmetic coding.
"""

import heapq
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import Tensor


def generate_histogram(
    delta: float = 0.645, dof: float = 5, n_samples: int = 2**18, seed: int = 100
) -> Tensor:
    """Returns a 256-bin histogram of scaled Student's t values."""
    torch.manual_seed(seed)
    samples = torch.distributions.StudentT(dof).sample((n_samples,))
    idx = samples.div(delta).round().int().clamp(-128, 127)
    return torch.bincount(idx + 128, minlength=256)


###########################################################################
# Huffman


def build_huffman_tree(histogram: Tensor) -> Dict[int, Tuple[int, int]]:
    # Filter out zero frequencies and create heap entries
    heap = []
    for symbol in range(len(histogram)):
        freq = histogram[symbol].item()
        if freq > 0:
            # Heap entries: (frequency, unique_id, node)
            # node is either (symbol,) for leaf or (left, right) for internal
            heapq.heappush(heap, (freq, symbol, (symbol,)))

    if not heap:
        return {}

    if len(heap) == 1:
        # Special case: single symbol
        freq, symbol, _ = heap[0]
        return {symbol: (0, 1)}

    # Build Huffman tree
    next_id = 256
    while len(heap) > 1:
        freq1, _, node1 = heapq.heappop(heap)
        freq2, _, node2 = heapq.heappop(heap)

        combined_freq = freq1 + freq2
        combined_node = (node1, node2)
        heapq.heappush(heap, (combined_freq, next_id, combined_node))
        next_id += 1

    # Extract codes from tree
    _, _, tree = heap[0]
    codes = {}

    def traverse(node, code, length):
        if isinstance(node, tuple) and len(node) == 1:
            # Leaf node
            symbol = node[0]
            codes[symbol] = (code, length)
        else:
            # Internal node
            left, right = node
            traverse(left, code << 1, length + 1)
            traverse(right, (code << 1) | 1, length + 1)

    traverse(tree, 0, 0)
    return codes


def build_canonical_huffman_codes(histogram: Tensor) -> Dict[int, Tuple[int, int]]:
    # Get non-canonical codes first
    codes_dict = build_huffman_tree(histogram)

    if not codes_dict:
        return {}

    # Sort by (length, symbol)
    sorted_symbols = sorted(codes_dict.items(), key=lambda x: (x[1][1], x[0]))

    # Assign canonical codes
    canonical_codes = {}
    code = 0
    prev_length = 0

    for symbol, (_, length) in sorted_symbols:
        if length > prev_length:
            code <<= length - prev_length
            prev_length = length
        canonical_codes[symbol] = (code, length)
        code += 1

    return canonical_codes


class CanonicalHuffmanDecoder:
    def __init__(self, codes: Dict[int, Tuple[int, int]], max_code_length: int = 16):
        self.codes = codes
        self.max_code_length = max_code_length
        self.decode_table = self._build_decode_table()

    def _build_decode_table(self) -> Dict[int, Tuple[int, int]]:
        table = {}
        for symbol, (code, length) in self.codes.items():
            # Pad code to max_code_length by left-shifting
            padded_code = code << (self.max_code_length - length)
            # Store for all possible bit patterns matching this prefix
            for suffix in range(1 << (self.max_code_length - length)):
                key = padded_code | suffix
                if key not in table:
                    table[key] = (symbol, length)
        return table

    def decode(self, bit_buffer: int, num_bits: int) -> List[int]:
        """Decode a sequence of symbols from a bit buffer."""
        symbols = []
        pos = 0

        while pos < num_bits:
            remaining = num_bits - pos

            # Extract the next max_code_length bits (or fewer if at end)
            bits_to_extract = min(self.max_code_length, remaining)

            # Extract bits starting at position pos
            bits = (bit_buffer >> (remaining - bits_to_extract)) & (
                (1 << bits_to_extract) - 1
            )

            # Pad extracted bits to max_code_length on the right for table lookup
            padded_bits = bits << (self.max_code_length - bits_to_extract)

            # Look up in the decode table using the padded bits
            symbol, length = self.decode_table.get(padded_bits, (-1, 0))

            if length == 0 or symbol == -1:
                raise ValueError(f"Failed to decode symbol at position {pos}")

            symbols.append(symbol)
            pos += length

        return symbols


class HuffmanEncoder:
    def __init__(self, codes: Dict[int, Tuple[int, int]]):
        self.codes = codes

    def encode(self, symbols: List[int]) -> Tuple[int, int]:
        """
        Returns a tuple of (bit_buffer, num_bits) representing the encoded data.
        """
        bit_buffer = 0
        num_bits = 0

        for symbol in symbols:
            if symbol not in self.codes:
                raise ValueError(f"Symbol {symbol} not in codebook")

            code, length = self.codes[symbol]
            bit_buffer = (bit_buffer << length) | code
            num_bits += length

        return bit_buffer, num_bits

    def encode_tensor(self, symbols: Tensor) -> Tuple[int, int]:
        return self.encode(symbols.tolist())


###########################################################################
# tANS


@dataclass
class TansDecodeEntry:
    symbol: int
    nb_bits: int
    new_x: int


def normalize_frequencies(histogram: Tensor, L: int) -> Dict[int, int]:
    total = histogram.sum().item()
    scale = L / total

    freqs = {}
    current_sum = 0

    # First pass: scale and ensure at least 1 for existing symbols
    # Sort by frequency to handle rounding errors gracefully
    sorted_indices = torch.argsort(histogram, descending=True)

    for idx in sorted_indices:
        symbol = idx.item()
        count = histogram[symbol].item()
        if count > 0:
            new_count = max(1, int(round(count * scale)))
            freqs[symbol] = new_count
            current_sum += new_count

    # Adjust sum to exactly L
    sorted_symbols = sorted(freqs.keys(), key=lambda s: freqs[s], reverse=True)

    if current_sum > L:
        diff = current_sum - L
        for s in sorted_symbols:
            if freqs[s] > 1:
                reduction = min(diff, freqs[s] - 1)
                freqs[s] -= reduction
                diff -= reduction
                if diff == 0:
                    break
    elif current_sum < L:
        diff = L - current_sum
        # Add to the most frequent symbol
        if sorted_symbols:
            freqs[sorted_symbols[0]] += diff

    return freqs


def build_tans_tables(
    histogram: Tensor, L: int = 2048
) -> Tuple[Dict[int, List[int]], List[TansDecodeEntry], Dict[int, int]]:
    freqs = normalize_frequencies(histogram, L)

    # Spread symbols
    symbol_spread = []
    for s, f in freqs.items():
        symbol_spread.extend([s] * f)

    # Deterministic shuffle
    rng = random.Random(42)
    rng.shuffle(symbol_spread)

    # Build Encode Table
    encode_table = {s: [] for s in freqs}

    # Build Decode Table
    decode_table = [None] * L

    # Track appearances
    next_appearance = {s: 0 for s in freqs}

    for i, s in enumerate(symbol_spread):
        x = L + i  # State in [L, 2L-1]

        # For encoding: map symbol's k-th appearance to state x
        encode_table[s].append(x)

        # For decoding
        k = next_appearance[s]
        next_appearance[s] += 1

        L_s = freqs[s]
        x_prev = L_s + k

        # Calculate nbBits
        nb_bits = L.bit_length() - x_prev.bit_length()
        if (x_prev << nb_bits) < L:
            nb_bits += 1

        new_x = x_prev << nb_bits

        decode_table[i] = TansDecodeEntry(symbol=s, nb_bits=nb_bits, new_x=new_x)

    return encode_table, decode_table, freqs


class TansEncoder:
    def __init__(
        self, freqs: Dict[int, int], encode_table: Dict[int, List[int]], L: int
    ):
        self.freqs = freqs
        self.encode_table = encode_table
        self.L = L

    def encode(self, symbols: List[int]) -> Tuple[List[int], int]:
        x = self.L
        bits = []

        for s in reversed(symbols):
            if s not in self.freqs:
                raise ValueError(f"Symbol {s} not in frequency table")
            L_s = self.freqs[s]
            while x >= 2 * L_s:
                bits.append(x & 1)
                x >>= 1

            idx = x - L_s
            x = self.encode_table[s][idx]

        return bits, x


class TansDecoder:
    def __init__(self, decode_table: List[TansDecodeEntry], L: int):
        self.decode_table = decode_table
        self.L = L

    def decode(self, bits: List[int], start_state: int, num_symbols: int) -> List[int]:
        x = start_state
        symbols = []
        bit_stack = list(bits)

        for _ in range(num_symbols):
            entry = self.decode_table[x - self.L]
            symbols.append(entry.symbol)

            nb_bits = entry.nb_bits
            read_val = 0
            for _ in range(nb_bits):
                if not bit_stack:
                    raise ValueError("Not enough bits")
                bit = bit_stack.pop()
                read_val = (read_val << 1) | bit

            x = entry.new_x + read_val

        return symbols


###########################################################################
# Arithmetic Coding


class ArithmeticCoder:
    def __init__(self, histogram: Tensor, precision: int = 32):
        self.precision = precision
        self.max_val = (1 << precision) - 1
        self.quarter = 1 << (precision - 2)
        self.half = 1 << (precision - 1)
        self.three_quarters = self.quarter * 3

        total = histogram.sum().item()
        self.total_freq = int(total)
        self.cum_freq = [0] * 257
        current = 0
        for i in range(256):
            self.cum_freq[i] = current
            current += int(histogram[i].item())
        self.cum_freq[256] = current

    def get_freq_range(self, symbol: int) -> Tuple[int, int]:
        return self.cum_freq[symbol], self.cum_freq[symbol + 1]

    def get_symbol_from_freq(self, value: int) -> Tuple[int, int, int]:
        # Find s such that cum_freq[s] <= value < cum_freq[s+1]
        l, r = 0, 256
        while l < r:
            mid = (l + r) // 2
            if self.cum_freq[mid + 1] <= value:
                l = mid + 1
            else:
                r = mid
        symbol = l
        return symbol, self.cum_freq[symbol], self.cum_freq[symbol + 1]


class ArithmeticEncoder(ArithmeticCoder):
    def encode(self, symbols: List[int]) -> List[int]:
        low = 0
        high = self.max_val
        bits = []
        pending_bits = 0

        for symbol in symbols:
            range_val = high - low + 1
            sym_low, sym_high = self.get_freq_range(symbol)

            high = low + (range_val * sym_high) // self.total_freq - 1
            low = low + (range_val * sym_low) // self.total_freq

            while True:
                if high < self.half:
                    bits.append(0)
                    bits.extend([1] * pending_bits)
                    pending_bits = 0
                elif low >= self.half:
                    bits.append(1)
                    bits.extend([0] * pending_bits)
                    pending_bits = 0
                    low -= self.half
                    high -= self.half
                elif low >= self.quarter and high < self.three_quarters:
                    pending_bits += 1
                    low -= self.quarter
                    high -= self.quarter
                else:
                    break

                low = 2 * low
                high = 2 * high + 1

        pending_bits += 1
        if low <= self.quarter:
            bits.append(0)
            bits.extend([1] * pending_bits)
        else:
            bits.append(1)
            bits.extend([0] * pending_bits)

        return bits


class ArithmeticDecoder(ArithmeticCoder):
    def decode(self, bits: List[int], num_symbols: int) -> List[int]:
        low = 0
        high = self.max_val
        value = 0

        bit_iterator = iter(bits)

        for _ in range(self.precision):
            value = value << 1
            try:
                value += next(bit_iterator)
            except StopIteration:
                pass

        symbols = []
        for _ in range(num_symbols):
            range_val = high - low + 1
            scaled_value = ((value - low + 1) * self.total_freq - 1) // range_val

            symbol, sym_low, sym_high = self.get_symbol_from_freq(scaled_value)
            symbols.append(symbol)

            high = low + (range_val * sym_high) // self.total_freq - 1
            low = low + (range_val * sym_low) // self.total_freq

            while True:
                if high < self.half:
                    pass
                elif low >= self.half:
                    value -= self.half
                    low -= self.half
                    high -= self.half
                elif low >= self.quarter and high < self.three_quarters:
                    value -= self.quarter
                    low -= self.quarter
                    high -= self.quarter
                else:
                    break

                low = 2 * low
                high = 2 * high + 1
                value = 2 * value
                try:
                    value += next(bit_iterator)
                except StopIteration:
                    pass

        return symbols


###########################################################################
# Drivers


def test_huffman() -> None:
    histogram = generate_histogram()
    histogram[histogram < histogram.sum() / 2**12] = 0  # Prune low-frequency symbols
    codes = build_canonical_huffman_codes(histogram)
    encoder = HuffmanEncoder(codes)
    decoder = CanonicalHuffmanDecoder(codes)

    test_symbols = [i for i in range(256) if histogram[i] > 0]
    bit_buffer, num_bits = encoder.encode(test_symbols)
    decoded_symbols = decoder.decode(bit_buffer, num_bits)

    print(f"Input: {test_symbols}")
    print(f"Output: {decoded_symbols}")
    assert decoded_symbols == test_symbols, "Decoded symbols do not match original"
    print("Huffman encoding/decoding test passed.")


def test_tans() -> None:
    histogram = generate_histogram()
    # Prune low-frequency symbols same as Huffman test for consistency
    histogram[histogram < histogram.sum() / 2**12] = 0

    L = 2048
    encode_table, decode_table, freqs = build_tans_tables(histogram, L)

    encoder = TansEncoder(freqs, encode_table, L)
    decoder = TansDecoder(decode_table, L)

    test_symbols = [i for i in range(256) if histogram[i] > 0]
    # Create a longer sequence to test compression
    test_sequence = []
    for s in test_symbols:
        test_sequence.extend([s] * 5)

    bits, final_state = encoder.encode(test_sequence)
    decoded_symbols = decoder.decode(bits, final_state, len(test_sequence))

    print(f"tANS Input length: {len(test_sequence)}")
    print(f"tANS Output bits: {len(bits)}")
    print(f"Compression ratio: {len(bits) / (len(test_sequence) * 8):.4f}")

    assert decoded_symbols == test_sequence, "Decoded symbols do not match original"
    print("tANS encoding/decoding test passed.")


def test_arithmetic() -> None:
    histogram = generate_histogram()
    # Prune low-frequency symbols same as Huffman test for consistency
    histogram[histogram < histogram.sum() / 2**12] = 0

    encoder = ArithmeticEncoder(histogram)
    decoder = ArithmeticDecoder(histogram)

    test_symbols = [i for i in range(256) if histogram[i] > 0]
    # Create a longer sequence to test compression
    test_sequence = []
    for s in test_symbols:
        test_sequence.extend([s] * 5)

    bits = encoder.encode(test_sequence)
    decoded_symbols = decoder.decode(bits, len(test_sequence))

    print(f"Arithmetic Input length: {len(test_sequence)}")
    print(f"Arithmetic Output bits: {len(bits)}")
    print(f"Compression ratio: {len(bits) / (len(test_sequence) * 8):.4f}")

    assert decoded_symbols == test_sequence, "Decoded symbols do not match original"
    print("Arithmetic encoding/decoding test passed.")


if __name__ == "__main__":
    test_huffman()
    print()
    test_tans()
    print()
    test_arithmetic()
    print()
