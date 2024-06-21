from utils import timed


class FailedToFindSolution(Exception):
    """Explicit error for when a function fails to find a solution (when it should, of course)."""


##########################################################################################
@timed
def problem_1():
    """Sum of all multiples of 3 or 5 that are below 1000."""
    # we can use the classic sum of ints equals n*(n+1)/2
    # but n is divided by the 'multiple of' value, and the result is multiplied by it
    # then we subtract our double counting
    last_number = 999

    def sum_of_mults(mult: int) -> int:
        occurrences = last_number // mult
        sum_of = occurrences * (occurrences + 1) * mult // 2
        return sum_of

    return sum_of_mults(3) + sum_of_mults(5) - sum_of_mults(15)


problem_1()


##########################################################################################
@timed
def problem_2():
    """Sum of even fibonacci numbers up to 4 million."""
    # even fibonacci numbers follow a+4b=c; instead of a+b=c
    # the first two even fibonacci numbers are 2,8
    a, b = 2, 8
    sum = a
    while b < 4e6:
        sum += b
        a, b = b, a + 4 * b
    return sum


problem_2()


##########################################################################################


@timed
def problem_3():
    """Find largest prime factor of 600,851,475,143"""
    n = 600851475143
    from primegen import FOUND_PRIMES, gen_next_prime

    pi = 0
    p = FOUND_PRIMES[pi]
    highest_prime = 0
    while n > 1:
        # while divisible, divide
        while n % p == 0:
            n /= p
            highest_prime = p
        # move to next prime
        pi += 1
        while pi >= len(FOUND_PRIMES):
            gen_next_prime()
        p = FOUND_PRIMES[pi]
    return highest_prime


problem_3()

##########################################################################################
# Largest Palindrome Product


@timed
def problem_4():
    """Find the largest palindrome made from the product of two 3-digit numbers."""
    import math

    # we will assume there is a 6-digit palindrome with two factors > sqrt(1e5)
    # we can construct palindromes directly to skip past useless numbers
    a, b, c = 9, 9, 9
    while True:
        # here are the digits as a palindrome
        palindrome = int(a * 1e5 + b * 1e4 + c * 1e3 + c * 1e2 + b * 1e1 + a * 1e0)
        # we will search for a factor below the square root - fewer numbers to check
        highest_possible_factor = math.floor(math.sqrt(palindrome))
        factors_invalid_from = palindrome // 1000
        for i in range(highest_possible_factor, factors_invalid_from, -1):
            # skip if not a factor or factor requires 4-digit number
            if palindrome % i > 0:
                continue
            return palindrome
        # generate the next palindrome by decrementing inner numbers, then outer numbers as needed
        c -= 1
        if c < 0:
            c = 9
            b -= 1
        if b < 0:
            b = 9
            a -= 1
        if a < 0:
            raise FailedToFindSolution()


problem_4()

##########################################################################################
# Smallest Multiple


@timed
def problem_5():
    """What is the smallest positive number that is EVENLY divisible by all the numbers from 1 to 20."""
    from primefactors import lowest_common_multiple

    # we do not have to muck around with "evenly divisible"
    # including 2 anywhere in the prime factors will guarantee that
    # and 2 itself guarantees 2 will be in the prime factors
    lcm = 2
    for i in range(2, 21):
        lcm = lowest_common_multiple(lcm, i)
    return lcm


problem_5()

##########################################################################################
# Sum Square Difference


@timed
def problem_6():
    """Find the difference between the sum of the squares of the first one hundred natural numbers and the square of the sum."""
    square_of_sum = (50 * 101) ** 2  # based on n*(n+1)/2 for n=100
    sum_of_squares = 0
    for i in range(1, 101):
        sum_of_squares += i**2
    diff = abs(sum_of_squares - square_of_sum)
    return diff


problem_6()

##########################################################################################
# 10001st Prime


@timed
def problem_7():
    """What is the 10,001st prime number?"""
    from primegen import FOUND_PRIMES, gen_next_prime

    num_to_gen = 10001 - len(FOUND_PRIMES)
    for _ in range(num_to_gen + 1):
        gen_next_prime()
    return FOUND_PRIMES[10000]


problem_7()

##########################################################################################
# Largest Product in a Series


@timed
def problem_8():
    """Find the thirteen adjacent digits in the $1000$-digit number that have the greatest product. What is the value of this product?"""
    import math

    with open("data/problem_8.txt", "r", encoding="utf-8") as filereader:
        digits = [int(digit) for digit in filereader.read() if digit not in (" ", "\n")]
    best_product = 0
    span_size = 13
    for left_index in range(0, len(digits) - span_size):
        right_index = left_index + span_size
        slice = digits[left_index:right_index]
        product = int(math.prod(slice))
        best_product = max(product, best_product)
    return best_product


problem_8()

##########################################################################################
# Special Pythagorean Triplet


@timed
def problem_9():
    """There exists exactly one Pythagorean triplet for which a + b + c = 1000. Find the product abc."""

    def is_triplet(a: int, b: int, c: int) -> bool:
        return a**2 + b**2 == c**2

    # start with c being a high value
    for c in range(998, 0, -1):
        # a starts low and works its way to (1 less than) 1000-c
        for a in range(1, 1000 - c):
            # b is whatever is left
            b = 1000 - c - a
            # test and return on success
            if is_triplet(a, b, c):
                return a * b * c
    raise FailedToFindSolution()


problem_9()

##########################################################################################
# Summation of Primes


@timed
def problem_10():
    """Find the sum of all the primes below two million."""
    from primegen import FOUND_PRIMES, gen_primes_to_n

    gen_primes_to_n(int(2e6))
    return sum(p for p in FOUND_PRIMES if p < 2e6)


problem_10()


##########################################################################################
# Largest Product in a Grid


@timed
def problem_11():
    """What is the greatest product of four adjacent numbers in the same direction (cardinal/diagonal) in the grid?"""
    # I forgot to do cardinals but fuckit
    with open("data/problem_11.txt", "r", encoding="utf-8") as filereader:
        grid = [[int(digits) for digits in line.split(" ")] for line in filereader.readlines()]
    best_product = 0
    # backslash
    backslash_offsets = [(0, 0), (1, 1), (2, 2), (3, 3)]
    forwardslash_offsets = [(3, 0), (2, 1), (1, 2), (0, 3)]
    vertical_offsets = [(0, 0), (0, 1), (0, 2), (0, 3)]
    horizontal_offsets = [(0, 0), (1, 0), (2, 0), (3, 0)]
    all_offsets = [backslash_offsets, forwardslash_offsets, vertical_offsets, horizontal_offsets]
    for left in range(20):
        for top in range(20):
            for offsets in all_offsets:
                product = 1
                for offset_col, offset_row in offsets:
                    col = left + offset_col
                    row = top + offset_row
                    if col >= 20 or row >= 20:
                        continue
                    product *= grid[row][col]
                best_product = max(best_product, product)
    return best_product


problem_11()

##########################################################################################
# Highly Divisible Triangular Number


@timed
def problem_12():
    """What is the value of the first triangle number to have over five hundred divisors?"""
    from primefactors import prime_factors_for, get_count_of_factors_from_prime_factors

    number, triangle_number = 0, 0
    while True:
        # advance triangle number
        number += 1
        triangle_number += number
        triangle_prime_factors = prime_factors_for(triangle_number)
        all_factors_count = get_count_of_factors_from_prime_factors(triangle_prime_factors)
        if all_factors_count > 500:
            return triangle_number


problem_12()

##########################################################################################
# Large Sum


@timed
def problem_13():
    """Work out the first ten digits of the sum of the following one-hundred 50-digit numbers"""
    # this is trivial in python
    with open("data/problem_13.txt", "r", encoding="utf-8") as filereader:
        numbers = [int(line) for line in filereader.readlines()]
    bigsum = sum(numbers)
    return int(str(bigsum)[:10])


problem_13()

##########################################################################################
# Longest Collatz Sequence


@timed
def problem_14():
    """Which starting number, under one million, produces the longest Collatz Sequence chain?"""
    from typing import Dict

    lengths: Dict[int, int] = {}

    def collatz_length_from(n: int):
        # already done just return it
        if n in lengths:
            return lengths[n]
        # build it
        ns = [n]
        length = 0
        while n > 1:
            if n % 2 == 0:
                n = n // 2
            else:
                n = 3 * n + 1
            # if we have seen this before we know the rest, otherwise keep building
            if n in lengths:
                length += lengths[n]
                break
            else:
                length += 1
                ns.append(n)
        # cache all the lengths we found
        for i, n in enumerate(ns):
            lengths[n] = length - i
        return length

    longest_length = 0
    longest_seed = 0
    for i in range(1_000_000, 0, -1):
        length = collatz_length_from(i)
        if length < longest_length:
            continue
        longest_length = length
        longest_seed = i
    return longest_seed


problem_14()

##########################################################################################
# Lattice Paths


@timed
def problem_15():
    """Starting in the top left of a 20x20 grid, and only being able to move right or down, how many possible routes are there to the bottom right?"""
    import math

    # must move 40 times, there are 20 decisions to make (which columns to move down in OR rows to move right in)
    return math.comb(40, 20)


problem_15()

##########################################################################################
# Power Digit Sum


@timed
def problem_16():
    """What is the sum of the digits of the number 2**1000?"""
    return sum(int(digit) for digit in str(2**1000))


problem_16()

##########################################################################################
# Number Letter Counts


@timed
def problem_17():
    """Writing out the words as text (1=one,2=two,etc); How many letter are written for the numbers 1 through 1000.
    Exclude hyphens and spaces. Include "and" joining."""
    # first entry is empty because if there are "zero" of this element we just do not write it at all
    units = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    # these two sets have slightly special naming conventions - note we do populate the "ten" rather than ""
    teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "INVALID", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
    # hundreds and above stop using special prefixes
    hundreds = ["", *[single + "hundred" for single in units[1:]]]
    thousands = ["", *[single + "thousand" for single in units[1:]]]

    def stringify(n: int) -> str:
        """Convert number to text, no spaces/dashes, includes 'and'"""
        # names these calculations to give some meaning
        index1000 = n // 1000
        mod1000 = n % 1000
        index100 = mod1000 // 100
        mod100 = n % 100
        mod10 = n % 10
        indexteen = mod100 - 10
        indexten = mod100 // 10
        indexunit = mod10
        # build the string
        text = ""
        text += thousands[index1000]
        text += hundreds[index100]
        text += "and" if n >= 100 and mod100 > 0 else ""
        # handle the special teens
        if 10 <= mod100 < 20:
            text += teens[indexteen]
        # otherwise it is simple
        else:
            text += tens[indexten]
            text += units[indexunit]
        return text

    return sum(len(stringify(n)) for n in range(1, 1000 + 1))


problem_17()

##########################################################################################
# Maximum Path Sum I


@timed
def problem_18():
    """By moving top-to-bottom, moving only one space left or right at a time, find the maximum sum attainable in the provided triangle."""
    with open("data/problem_18.txt", "r", encoding="utf-8") as filereader:
        triangle = [[int(digits) for digits in line.split(" ")] for line in filereader.readlines()]
    # faster to just roll up from bottom
    triangle.reverse()
    for i in range(1, len(triangle)):
        for j in range(len(triangle[i])):
            triangle[i][j] += max(triangle[i - 1][j : j + 2])
    return triangle[-1][0]


problem_18()

##########################################################################################
# Counting Sundays


@timed
def problem_19():
    """How many Sundays fell on the first of the month during the twentieth century (1 Jan 1901 to 31 Dec 2000)?"""
    # will do this with algo rather than datetime library
    # we will assert that Sunday is 0th day of week
    # and that 1900-01-01 was Monday (2)
    year = 1901
    month_of_year = 0  # zero-indexed
    day_of_week = (2 + 365) % 7  # monday for 1900-01-01, skip to 1901, not a leap year
    sundays = 0
    month_lengths = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    # for readability
    SUNDAY = 0
    JANUARY = 0
    FEBRUARY = 1
    while year < 2001:
        # if we are on a sunday add to counter
        sundays += 1 if day_of_week == SUNDAY else 0
        # advance time
        move_days = month_lengths[month_of_year]
        move_days += 1 if month_of_year == FEBRUARY and (year % 400 == 0 or (year % 100 != 0 and year % 4 == 0)) else 0
        day_of_week = (day_of_week + move_days) % 7
        month_of_year = (month_of_year + 1) % 12
        year += 1 if month_of_year == JANUARY else 0
    return sundays


problem_19()


# and using datetime library
@timed
def problem_19a():
    from datetime import date

    sundays = 0
    for year in range(1901, 2001):
        for month in range(1, 13):
            sundays += 1 if date(year, month, 1).weekday() == 6 else 0
    return sundays


problem_19a()


##########################################################################################
# Factorial Digit Sum


@timed
def problem_20():
    """Find the sum of the digits in the number 100!"""
    import math

    return sum(int(d) for d in str(math.factorial(100)))


problem_20()

##########################################################################################
# Amicable Numbers


@timed
def problem_21():
    """Evaluate the sum of all the amicable numbers under 10000"""
    from primefactors import all_factors_for

    amicable_cache = [0] * 10000
    for i in range(1, 10000):
        amicable_cache[i] = sum(all_factors_for(i)) - i
    sum_of_amicable = sum(
        i
        for i in range(1, 10000)
        # avoid index out of range
        if amicable_cache[i] < 10000
        # not perfect numbers
        and amicable_cache[i] != i
        # amicable ones
        and amicable_cache[amicable_cache[i]] == i
    )
    return sum_of_amicable


problem_21()

##########################################################################################
# Names Scores


@timed
def problem_22():
    """What is the total of all the name scores in the file?
    Letters score based on position in alphabet, A=1
    Names multiply score based on position in list, 1-indexed"""
    letter_score = {c: i + 1 for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
    with open("data/problem_22.txt", "r", encoding="utf-8") as filereader:
        text = filereader.read()
    names = text.replace('"', "").split(",")
    names.sort()
    total_score = sum(sum(letter_score[c] for c in name) * (i + 1) for i, name in enumerate(names))
    return total_score


problem_22()

##########################################################################################
# Non-Abundant Sums


@timed
def problem_23():
    """Find the sum of all the positive integers which cannot be written as the sum of two abundant numbers.
    Everything above 28123 DEFINITELY can be the sum of two abundant numbers."""
    from primefactors import all_factors_for

    limit = 28123
    # find which numbers are abundant -- anything beyond the limit is useless as per the problem information
    # the list here is ordered so that we can short-circuit knowing all following options must be false
    abundants = [i for i in range(1, limit) if sum(all_factors_for(i)) > 2 * i]
    # this is a quick lookup to determine if a given number is abundant
    abundant_set = {i for i in abundants}

    # go through each number up to the limit, check if it minus any abundant gives another abundant
    def is_sum_of_two_abundants(n: int) -> bool:
        search_limit = n // 2
        for a in abundants:
            # did not find one before the abundant numbers got too big
            if a > search_limit:
                return False
            # n-a=b is.same.as a+b=n
            if (n - a) in abundant_set:
                return True
        return False

    rolling_sum = 0
    for i in range(1, limit):
        if not is_sum_of_two_abundants(i):
            rolling_sum += i
    return rolling_sum


problem_23()

##########################################################################################
# Lexicographic Permutations


@timed
def problem_24():
    """What is the millionth lexicographic permutation of the digits 0, 1, 2, 3, 4, 5, 6, 7, 8 and 9?"""
    from itertools import permutations

    for i, x in enumerate(permutations([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])):
        if i >= 999999:
            return int("".join(str(d) for d in x))
    raise FailedToFindSolution()


problem_24()

##########################################################################################
# 1000-digit Fibonacci Number


@timed
def problem_25():
    """What is the index of the first term in the Fibonacci sequence to contain 1000 digits?"""
    a, b = 1, 1
    limit = 10**999
    i = 2  # starting at 2nd term
    while b < limit:
        a, b = b, a + b
        i += 1
    return i


problem_25()

##########################################################################################
# Reciprocal Cycles


@timed
def problem_26():
    """Find the value of d<1000 for which 1/d contains the longest recurring cycle in its decimal fraction part."""
    from primegen import gen_primes_to_n, FOUND_PRIMES

    # we will need primes up to our limit
    limit = 1000
    gen_primes_to_n(limit)

    # simulate long division until we find the first digit we started with, then it should cycle
    def cycle_length(divisor: int) -> int:
        # we are doing 1/x so we always start with remainder 1 then carry
        x = 10
        length = 0
        while True:
            x = x % divisor
            length += 1
            if x == 1:
                return length
            x *= 10

    # find the longest value
    best = 0, 0
    for p in FOUND_PRIMES:
        if p % 2 == 0 or p % 5 == 0:
            continue
        if p > 1000:
            break
        l = cycle_length(p)
        if l > best[1]:
            best = p, l
    return best[0]


problem_26()

##########################################################################################
# Quadratic Primes


@timed
def problem_27():
    """Find the product of the coefficients A and B,
    for the quadratic expression N^2 + NA + B,
    that produces the maximum number of primes for consecutive values of N starting from 0
    Restrained to abs(A) < 1000, abs(B) <= 1000."""
    from primegen import FOUND_PRIMES, gen_primes_to_n

    gen_primes_to_n(1000)
    # since n^2+an=0 for n=0, b must be a prime number
    b_values = [p for p in FOUND_PRIMES if p < 1000]
    # expand our primes
    gen_primes_to_n(1_000_000)
    # make a fast search
    prime_search = set(FOUND_PRIMES)
    # track our best result
    best = 0, 0
    for a in range(-1000, 1000):
        for b in b_values:
            i = 0
            while True:
                v = i**2 + a * i + b
                if v > FOUND_PRIMES[-1]:
                    gen_primes_to_n(v)
                    prime_search = set(FOUND_PRIMES)
                if v not in prime_search:
                    break
                i += 1
            if i > best[1]:
                best = a * b, i
    return best[0]


problem_27()


##########################################################################################
# Number Spiral Diagonals


@timed
def problem_28():
    """What is the sum of the numbers on the diagonals in a 1001 by 1001 spiral
    formed with sequential number spiraling clockwise outwards, starting moving right?"""
    size = 1
    value = 1
    last = 1
    # corners are a consistent distance from last corner based on size
    while size < 1001:
        size += 2
        steps = size - 1
        a = last + steps
        b = a + steps
        c = b + steps
        last = c + steps
        value += a + b + c + last
    return value


problem_28()

##########################################################################################
# Distinct Powers


@timed
def problem_29():
    """How many distinct terms are in the sequence generated by a**b for 2<=a<=100 and 2<=b<=100"""
    # there are only 10k combinations, easy to put into a set
    terms = set()
    for a in range(2, 101):
        for b in range(2, 101):
            terms.add(a**b)
    return len(terms)


problem_29()

##########################################################################################
# Digit Fifth Powers


@timed
def problem30():
    """Find the sum of all the numbers that can be written as the sum of fifth powers of their digits."""
    import itertools
    from digits import number_to_digits

    # calculate the fifth powers once for each digit
    fifth_powers = [i**5 for i in range(10)]
    # eventually the number will outstrip the ability of sum to keep up - this finds the upper limit
    # we subtract 1 because if we did have that many digits, we could not reach the number
    limit = max(len(str(i * fifth_powers[9])) for i in range(100) if len(str(i * fifth_powers[9])) <= i) - 1
    # the sum of digits is the same regardless of order so just find all the distinct combinations of digits up to digit length
    distinct_digits = [i for i in itertools.combinations_with_replacement([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], limit)]
    # calculate each sum just once
    sums_of_fifth_powers_of_digits = [sum(fifth_powers[d] for d in digits) for digits in distinct_digits]
    # now we just need to check each distinct combination to see if its corresponding sum matches
    sum_of_sums = 0
    for i in range(1, len(distinct_digits)):
        value = sums_of_fifth_powers_of_digits[i]
        value_as_digits = number_to_digits(value)
        # we will use the fact that combinations are ordered to just match up the digits
        comparable = tuple(sorted(value_as_digits + [0] * (6 - len(value_as_digits))))
        if comparable == distinct_digits[i]:
            sum_of_sums += value
    # `1` does not count because it is not a sum of 2 numbers
    sum_of_sums -= 1
    return sum_of_sums


problem30()


##########################################################################################
# Coin Sums


@timed
def problem_31():
    """How many different ways can £2 be made using any number of coins?"""
    # ie how to make 200 from combinations of 200, 100, 50, 20, 10, 5, 2, 1
    # dynamic programming will work
    # array of ways to hit each value;
    # for each coin add to ways to hit this value, ways we could hit this value - coin value
    # we start with 1 way to have a value of 0 (no coins)
    ways = [0] * 201
    ways[0] = 1
    for v in [1, 2, 5, 10, 20, 50, 100, 200]:
        for i in range(v, 201):
            ways[i] += ways[i - v]
    return ways[200]


problem_31()

##########################################################################################
# Pandigital Products


@timed
def problem_32():
    """Find the sum of all products whose multiplicand/multiplier/product identity can be written as a 1-9 pandigital.
    That is for a*b=c, all digits 1 through 9 appear exactly once across all three numbers."""

    from itertools import permutations

    # we will accumulate unique products here
    valid_products = set()
    # iterate over permutations
    # we use numbers directly so that we do not waste time in string manipulation and integer parsing
    for d in permutations([1, 2, 3, 4, 5, 6, 7, 8, 9]):
        # we only need to calc c once in this permutation, it must be 4 digits
        c = d[5] * 1000 + d[6] * 100 + d[7] * 10 + d[8]
        # if we saw c as a valid product we can skip
        if c in valid_products:
            continue
        # there are only a few combinations of digit lengths that can work: 1,4; 2,3;
        # c always has some or more digits than higher of a,b;
        # therefore c must be at least 4 digits with 1,4 and 2,3 splits
        # we can exclude 3,3,3 splits because the lowest possible value is 100*100=10000 which is a total of 11 digits
        # since a*b=c and b*a=c, we only care about c, and a and b have different lengths, we can assert a is the smaller value
        # also, we have unrolled the calculations to avoid having to calculate loops so much
        # 1,4
        a = d[0]
        b = d[1] * 1000 + d[2] * 100 + d[3] * 10 + d[4]
        if a * b == c:
            valid_products.add(c)
            continue
        # 2,3
        a = d[0] * 10 + d[1]
        b = d[2] * 100 + d[3] * 10 + d[4]
        if a * b == c:
            valid_products.add(c)
    return sum(valid_products)


problem_32()


##########################################################################################
# Digit Cancelling Fractions


@timed
def problem_33():
    """There are four non-trivial examples of fractions<1 where cancelling a common digit
    maintains the value. Multiply these fractions together, simplify to lowest common terms,
    and provide that denominator"""

    from primefactors import simplify_fraction
    import math

    # quick int→str conversion
    s = [str(n) for n in range(100)]
    # accumulator
    found_items = []

    def advancing_pairs(min: int, max_exclusive: int):
        for i in range(min, max_exclusive):
            for j in range(i + 1, max_exclusive):
                yield i, j

    for i, j in advancing_pairs(11, 100):
        if i % 10 == 0 or j % 10 == 0:
            continue
        fraction = i / j
        si = s[i]
        sj = s[j]
        # only works if we remove tens from numerator and units from denominator
        if si[1] == sj[0]:
            n, d = int(si[0]), int(sj[1])
            if n / d == fraction:
                found_items.append((n, d))

    n = math.prod(n for n, d in found_items)
    d = math.prod(d for n, d in found_items)
    n, d = simplify_fraction(n, d)
    return d


problem_33()

##########################################################################################
# Digit Factorials


@timed
def problem_34():
    """Find the sum of all numbers which are equal to the sum of the factorial of their digits."""
    import math
    import itertools

    # pre-calculate the digit factorials - it isn't like they are going to change
    digit_factorials = [math.factorial(n) for n in range(10)]
    accumulator = 0
    # experimentally 3,5 were the only depths required : validated by running 2,3,4,5,6 and getting correct result from only 3,5 numbers;
    for depth in (3, 5):
        # we are using some slightly cached values to speed up the inner loop
        # we only calculate the place powers once per depth, we will multiply with digits and get sum to produce value
        placepowers = tuple(10**n for n in range(depth - 1, -1, -1))
        # the range object is only created once for the depth, and reused for each combination
        depth_range = range(depth)
        for digits in itertools.product([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], repeat=depth):
            # leading zero breaks the algo since 0!==1
            if digits[0] == 0:
                continue
            # calculate the value of the number with these digits, and the sum of its digit factorials
            base_value = sum(placepowers[i] * digits[i] for i in depth_range)
            factorial_value = sum(digit_factorials[digit] for digit in digits)
            #
            if base_value == factorial_value:
                # OPTIONAL print(f"At depth {depth} the sum of digit factorials for {base_value} is {factorial_value}")
                accumulator += base_value
    return accumulator


problem_34()

##########################################################################################
# Circular Primes


@timed
def problem_35():
    """How many circular primes are there below one million?
    A circular prime is one where every rotation of its digits is also prime.
    Note that is 'rotation' NOT 'permutation'."""
    import itertools
    from primegen import yield_primes
    from digits import number_to_digits, digit_rotator, digits_to_number

    # build list/sets for quickly referring to values
    checkable_primes = set(itertools.takewhile(lambda x: x < 1_000_000, yield_primes()))
    already_done = set()
    # the single digits do not provide a rotation to check so we add them as a special case
    rotatable_primes = {2, 3, 5, 7}
    # any number with any of these digits is invalid, because at least one rotation will be a multiple of 2 or 5
    breaking_digits = {2, 4, 5, 6, 8, 0}
    for p in checkable_primes:
        # skip if we have seen this already
        if p in already_done:
            continue
        # go through all rotations to see if they are all prime
        digits = number_to_digits(p)
        # skip if any digit would cause a rotation to be multiple of 2 or 5
        if any(d in breaking_digits for d in digits):
            continue
        rotated_values = [digits_to_number(rotation) for rotation in digit_rotator(digits)]
        # mark all rotations as checked so that when we hit the other primes (if any) we do not process them
        for x in rotated_values:
            already_done.add(x)
        # if all rotated values are prime we can extend the found rotatable primes
        if all(x in checkable_primes for x in rotated_values):
            for x in rotated_values:
                rotatable_primes.add(x)
    return len(rotatable_primes)


problem_35()

##########################################################################################
# Double-base Palindromes


@timed
def problem_36():
    """Find the sum of all numbers, less than one million, which are palindromic in base 10 and base 2.
    (No leading zeroes)"""
    from digits import generate_palindrome_digits, digits_to_number

    accumulator = 0
    # we will generate base10 palindromes directly as list of digits
    for size in range(1, 7):
        for palindrome_digits in generate_palindrome_digits(size):
            # even numbers are always 1..0 in binary
            if palindrome_digits[0] % 2 == 0:
                continue
            # convert to actual number, then to binary string, then compare against reversed string
            n = digits_to_number(palindrome_digits)
            b = f"{n:b}"
            if b != b[::-1]:
                continue
            # all checks passed
            accumulator += n
    return accumulator


problem_36()

##########################################################################################
# Truncatable Primes


@timed
def problem_37():
    """Find the sum of the only eleven primes that are both truncatable from left to right and right to left.
    This excludes single-digit primes"""

    from primegen import FOUND_PRIMES, gen_next_prime
    import math

    counter = 0
    accumulator = 0
    prime_lookup = set(FOUND_PRIMES)
    # all digits
    prime_index = 3  # skipping 2,3,5,7
    while counter < 11:
        prime_index += 1
        while prime_index >= len(FOUND_PRIMES):
            gen_next_prime()
            prime_lookup.add(FOUND_PRIMES[-1])
        p = FOUND_PRIMES[prime_index]
        range_ = range(1, math.floor(math.log10(p)) + 1)
        if all(p % (10**i) in prime_lookup and p // (10**i) in prime_lookup for i in range_):
            counter += 1
            accumulator += p
    return accumulator


problem_37()

##########################################################################################
# Pandigital Multiples


@timed
def problem38():
    """What is the largest pan-digital number (all digits 1-9) that can be created by some number,
    and concatenating the results of multiplying it by 1..n, for n>1."""
    best = 0
    # we could trim this up with expectations, but this is an exhaustive search
    for n in range(2, 10):
        for base in range(10000):
            stringy = "".join(str(base * i) for i in range(1, n + 1))
            # must be 9 digits
            if len(stringy) != 9:
                continue
            # cannot have a 0
            if "0" in stringy:
                continue
            # must be unique digits
            if len(set(stringy)) != 9:
                continue
            # candidate, but we need to check values after
            value = int(stringy)
            best = max(best, value)
    return best


problem38()
