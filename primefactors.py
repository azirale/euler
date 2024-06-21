from typing import List, Dict, Tuple
import math
from primegen import FOUND_PRIMES, gen_next_prime

_PRIME_FACTOR_CACHE: Dict[int, List[int]] = {}


def prime_factors_for(n: int) -> List[int]:
    # we remember these values because we might be looping over many numbers to find common factors
    if n in _PRIME_FACTOR_CACHE:
        return _PRIME_FACTOR_CACHE[n]
    # if we have not done this number previously then calculate it
    prime_index = 0
    prime = FOUND_PRIMES[prime_index]
    prime_factors = []
    calculated_for_ns = [n]
    while n > 1:
        # if this n is already cached we can short-circuit
        if n in _PRIME_FACTOR_CACHE:
            prime_factors.extend(_PRIME_FACTOR_CACHE[n])
            break
        # while divisible, divide, and track other things we are finding factors for
        while n % prime == 0:
            n = n // prime
            prime_factors.append(prime)
            calculated_for_ns.append(n)
        # move to next prime
        prime_index += 1
        while prime_index >= len(FOUND_PRIMES):
            gen_next_prime()
        prime = FOUND_PRIMES[prime_index]
    # now that the loop is done we know all the prime factors for this n
    # we also know the prime factors for each division we did - we just go by index order
    # since we already calculated them, just include them
    for i in range(len(calculated_for_ns)):
        _PRIME_FACTOR_CACHE[calculated_for_ns[i]] = prime_factors[i:]
    return _PRIME_FACTOR_CACHE[calculated_for_ns[0]]


def prime_factors_for_lowest_common_multiple(a: int, b: int) -> List[int]:
    a_prime_factors = prime_factors_for(a)
    b_prime_factors = prime_factors_for(b)
    distinct_prime_factors = set(a_prime_factors) | set(b_prime_factors)
    lcm_prime_factors: List[int] = []
    for prime_factor in distinct_prime_factors:
        a_count = a_prime_factors.count(prime_factor)
        b_count = b_prime_factors.count(prime_factor)
        higher_count = max(a_count, b_count)
        lcm_prime_factors.extend([prime_factor] * higher_count)
    return lcm_prime_factors


def lowest_common_multiple(a: int, b: int) -> int:
    lcm_prime_factors = prime_factors_for_lowest_common_multiple(a, b)
    product_of = product_over_prime_factors(lcm_prime_factors)
    return product_of


def product_over_prime_factors(prime_factors: List[int]) -> int:
    return int(math.prod(prime_factors))


def expand_prime_factors_to_all_factors(prime_factors: List[int]) -> List[int]:
    primes = set(prime_factors)
    prime_counts = {p: prime_factors.count(p) for p in primes}
    factors = [1]
    for prime, count in prime_counts.items():
        new_factors = [factor * prime**e for factor in factors for e in range(1, count + 1)]
        factors.extend(new_factors)
    return factors


def get_count_of_factors_from_prime_factors(prime_factors: List[int]) -> int:
    primes = set(prime_factors)
    prime_counts = {p: prime_factors.count(p) for p in primes}
    factor_count = 1
    for count in prime_counts.values():
        factor_count += factor_count * count
    return factor_count


def all_factors_for(n: int) -> List[int]:
    prime_factors = prime_factors_for(n)
    all_factors = expand_prime_factors_to_all_factors(prime_factors)
    return all_factors


def cancel_common_factors(a_prime_factors: List[int], b_prime_factors: List[int]) -> Tuple[List[int], List[int]]:
    a_index, b_index = 0, 0
    a_prime_factors.sort()
    b_prime_factors.sort()
    new_a = []
    new_b = []
    while a_index < len(a_prime_factors) and b_index < len(b_prime_factors):
        # cancelled
        if a_prime_factors[a_index] == b_prime_factors[b_index]:
            a_index += 1
            b_index += 1
            continue
        # if a>b then add b and advance b track
        if a_prime_factors[a_index] > b_prime_factors[b_index]:
            new_b.append(b_prime_factors[b_index])
            b_index += 1
            continue
        # if a<b then add a and advance a track
        if a_prime_factors[a_index] < b_prime_factors[b_index]:
            new_a.append(a_prime_factors[a_index])
            a_index += 1
            continue
    # extend anything that was skipped because other finished first
    new_a.extend(a_prime_factors[a_index:])
    new_b.extend(b_prime_factors[b_index:])
    return new_a, new_b


def simplify_fraction(numerator: int, denominator: int) -> Tuple[int, int]:
    # find the prime factors for numerator/denominator
    npf, dpf = prime_factors_for(numerator), prime_factors_for(denominator)
    # factors that are not common between them -- ie will generate simplified fraction values
    nsf, dsf = cancel_common_factors(npf, dpf)
    # the actuals implified fraction values, rather than the prime factors
    ns, ds = math.prod(nsf), math.prod(dsf)
    return ns, ds
