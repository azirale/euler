from typing import Dict, Tuple, List
from bisect import bisect_right
import timeit
import math

# we will report how long it takes to init
_init_start = timeit.default_timer()
print("Initialising prime generator...")

# we will sequentially find more primes and add to this list in ascending order
# we seed this with the first two primes
FOUND_PRIMES = [2, 3]


def have_reached_primes_up_to_n(n: int) -> int:
    """Have we found all the primes up to some value?"""
    return n <= FOUND_PRIMES[-1]


def have_found_first_n_primes(n: int) -> int:
    """Have we found this many primes?"""
    return len(FOUND_PRIMES)


def gen_primes_to_n(n: int) -> None:
    while not have_reached_primes_up_to_n(n):
        gen_next_prime()


# this is used for generating primes
# we will come up with an array of gaps to iterate through
# that lets us add by ~5 at a time
_GAPS = [2, 4]

# with our gap generator the first prime to start building gaps from will be 5
# the seed gaps already incorporate 2 and 3
_GAP_FROM = 5

# once we start generating we will need to move through gap list and loop around
_GAP_INDEX = 0

# then we want a quick way to test if a number is not prime
# in our construction we know what the NEXT number to be knocked by each prime is
# so we can put that in a dict to see if our N is in the dict
# then we can delete it and set the key up by the value
_QUICK_SKIPS: Dict[int, List[List[int]]] = {}

# GENERATE GAPS
while len(_GAPS) < 1e6:
    # we are going to incorporate the gap starting value into
    FOUND_PRIMES.append(_GAP_FROM)
    modulo_gaps = []
    modulo_rowtotal = 0
    for gap in _GAPS:
        modulo_rowtotal += gap
        modulo_rowtotal = modulo_rowtotal % _GAP_FROM
        modulo_gaps.append(modulo_rowtotal)
    new_gaps = []
    carry = 0
    # r represents the index for each row where each row is a full copy of the previous gap list
    for r in range(_GAP_FROM):
        # c represents the index for each column, that being a position along the previous gap list
        for c in range(len(_GAPS)):
            # we skip the first entry because we will 'consume' it to get to the next _NEXT_GAP_PRIME
            if r == 0 and c == 0:
                continue
            this_modulo = (modulo_gaps[c] + (modulo_rowtotal * r)) % _GAP_FROM
            # when the modulo at this row-by-column is 0 then we have a multiple of _NEXT_GAP_PRIME -- we carry this gap value into the next one
            if this_modulo == 0:
                carry = _GAPS[c]
            # if the modulo is not zero this number is not divisible by any number up to _NEXT_GAP_PRIME, add its gap and reset the carry
            else:
                new_gaps.append(_GAPS[c] + carry)
                carry = 0
    # we ignored the first skip since it would be consumed but it is still part of the sequence
    # add it back as the last skip as if we rotated the sequence positions and include the last carry value
    new_gaps.append(_GAPS[0] + carry)
    # update initial conditions for next iteration
    # use the first skip of the previous skip list to get next prime
    _GAP_FROM += _GAPS[0]
    # replace the original skip list with the new derived one for the new start prime
    _GAPS = new_gaps

print(f"Gaps will avoid primes = {FOUND_PRIMES}")
print(f"The gap sequence is {len(_GAPS):,} entries long")
print(f"Composite skips will start from {_GAP_FROM}, and then advance to each prime")

_FIRST_GAP_FROM = _GAP_FROM
_SEMIPRIME_HITS = 0


# this is the core prime generator
# we check to see if GAP_FROM is in _SKIPS, if so add the gap to gap from and try again
# once GAP_FROM is not in _SKIPS, that is the next prime, add it to primes and add gap to gapfrom
def gen_next_prime():
    # most composites will be semiprimes, and we can track those easily
    while _GAP_FROM in _QUICK_SKIPS:
        # get the underlying prime factors that caused this skip
        prime_factors_lists = _QUICK_SKIPS[_GAP_FROM]
        # remove this skip - we have moved past it
        del _QUICK_SKIPS[_GAP_FROM]
        for prime_factors in prime_factors_lists:
            # advance last factor to next prime
            next_pfs = [p for p in prime_factors]
            drop_p = prime_factors[-1]
            next_p = FOUND_PRIMES[bisect_right(FOUND_PRIMES, drop_p)]
            next_pfs[-1] = next_p
            next_composite = _GAP_FROM // drop_p * next_p
            if next_composite not in _QUICK_SKIPS:
                _QUICK_SKIPS[next_composite] = []
            _QUICK_SKIPS[next_composite].append(next_pfs)
            # deepen prime factors
            deeper_pfs = [p for p in prime_factors]
            deeper_pfs.append(_FIRST_GAP_FROM)
            deeper_composite = int(math.prod(deeper_pfs))
            if deeper_composite not in _QUICK_SKIPS:
                _QUICK_SKIPS[deeper_composite] = []
            _QUICK_SKIPS[deeper_composite].append(deeper_pfs)
        # we also need to advance the gap
        global _SEMIPRIME_HITS
        _SEMIPRIME_HITS += 1
        advance_gap()
    # once we are out of the skip loop then we have found a prime
    FOUND_PRIMES.append(_GAP_FROM)
    # we also need to add its first skip, which is its own square
    _QUICK_SKIPS[_GAP_FROM**2] = [[_GAP_FROM, _GAP_FROM]]
    # and advance the gap
    advance_gap()


def advance_gap():
    global _GAP_FROM, _GAP_INDEX
    _GAP_FROM += _GAPS[_GAP_INDEX]
    _GAP_INDEX = (_GAP_INDEX + 1) % len(_GAPS)


def yield_primes():
    i = 0
    while True:
        yield FOUND_PRIMES[i]
        i += 1
        if i == len(FOUND_PRIMES):
            gen_next_prime()


def _BENCHMARK():
    numgen = 1000000
    s = timeit.default_timer()
    for i in range(numgen):
        gen_next_prime()
    e = timeit.default_timer()
    print(f"Generated {numgen} primes in {e-s:.2f}s")


def _TEST_GENERATED_PRIMES():
    for p in FOUND_PRIMES[1:]:
        if p % 2 == 0:
            print(f"ffs {p} is an even number")
        for i in range(3, p, 2):
            if p % i == 0:
                j = p // i
                print(f"FAILED on {p} -- it is {i} * {j}")
                return
    print("PASSED")


_init_end = timeit.default_timer()
print(f"Primegen initialisation done in {_init_end-_init_start:.2}s")
print("----------------------------------------")
