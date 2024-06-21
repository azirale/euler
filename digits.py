"""For when the core of a problem involves working with digits, and it is faster
to iterate over lists of digits than converting to string, splitting, and re-parsing."""

from functools import lru_cache
import itertools
from typing import List, Generator, Tuple, Union
import math

Digits = List[int]
"""A list representing the digits of a number. For example `123` is represented as `[1,2,3]`."""

Powers = List[int]
"""A list representing the powers of 10 to multiply each digit of Digits to, to get its place value."""


@lru_cache
def get_powertools(num_digits: int) -> Tuple[range, Powers]:
    """Memoised copy of range generator and `Powers` for a given number of digits.
    Memoisation helps save time in tight loops, even when number of digits is not known ahead of time."""
    _range = range(num_digits)
    _powers = [10 ** (num_digits - j - 1) for j in _range]
    return _range, _powers


def digits_to_number(digits: Digits) -> int:
    digit_range, digit_powers = get_powertools(len(digits))
    accumulator = sum(digits[i] * digit_powers[i] for i in digit_range)
    return accumulator


def all_digit_rotations(digits: Digits) -> List[Digits]:
    """Provides all rotations of"""
    length = len(digits)
    rotatable_digits = digits + digits
    digit_rotations = [rotatable_digits[i : i + length] for i in range(length)]
    return digit_rotations


def digit_rotator(digits: Digits) -> Generator[Digits, None, None]:
    """Generates rotations of digits, until all rotations are provided."""
    # first goes as-is
    yield digits
    # subsequent values take two slices
    for i in range(1, len(digits)):
        yield digits[i:] + digits[:i]


def number_to_digits(n: int) -> Digits:
    num_digits = math.floor(math.log10(n)) + 1
    digit_range, digit_powers = get_powertools(num_digits)
    digits = [(n // (digit_powers[i]) % 10) for i in digit_range]
    return digits


def generate_palindrome_digits(num_digits: int) -> Generator[Digits, None, None]:
    # outer digits cannot have a (leading) zero
    outer_digits = [d for d in range(1, 10)]
    inner_digits = [d for d in range(10)]
    # special case for single digit - has no reflected digits
    if num_digits == 1:
        for i in outer_digits:
            yield [i]
        return
    # handles the reverse slicing to exclude nonexistent 'middle' digit for an even number of digits
    slicer = slice(None, None, -1) if num_digits % 2 == 0 else slice(-2, None, -1)
    for outer in outer_digits:
        for inners in itertools.product(inner_digits, repeat=(num_digits - 1) // 2):
            yield [outer, *inners, *inners[slicer], outer]
