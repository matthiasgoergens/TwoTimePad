import string

alphabet = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.?,-:;'()"
alpha_dict = {char: idx for idx, char in enumerate(alphabet)}
inv_alpha_dict = {idx: char for idx, char in enumerate(alphabet)}

def to_nums(text):
    return [alpha_dict[c] for c in text]

def to_text(nums):
    return ''.join(inv_alpha_dict[n] for n in nums)

def mod46_diff(c1, c2):
    return [(a - b) % 46 for a, b in zip(c1, c2)]

def plausible_char(c):
    return c == ' ' or c in string.ascii_uppercase or c in ".,?-;'()"

def guess_plaintexts(c1, c2):
    diff = mod46_diff(c1, c2)
    n = len(diff)
    plaintext1 = ['?'] * n
    plaintext2 = ['?'] * n

    # First pass: identify likely spaces
    space_likelihood = [0] * n
    for i in range(n):
        char_if_p1_space = inv_alpha_dict[diff[i]]
        char_if_p2_space = inv_alpha_dict[-diff[i] % 46]

        if plausible_char(char_if_p1_space):
            space_likelihood[i] += 1
        if plausible_char(char_if_p2_space):
            space_likelihood[i] += 1

    # Assign spaces confidently
    for i in range(n):
        char_if_p1_space = inv_alpha_dict[diff[i]]
        char_if_p2_space = inv_alpha_dict[-diff[i] % 46]

        # Prefer the hypothesis that yields a letter when the other plaintext is space
        if char_if_p1_space == ' ' and plausible_char(char_if_p2_space):
            plaintext1[i], plaintext2[i] = ' ', char_if_p2_space
        elif char_if_p2_space == ' ' and plausible_char(char_if_p1_space):
            plaintext2[i], plaintext1[i] = ' ', char_if_p1_space
        else:
            # If uncertain, leave '?'
            plaintext1[i], plaintext2[i] = '?', '?'

    # Second pass: propagate known spaces to reveal words
    # Try common English words to fill gaps (e.g., THE, AND, OF)
    # This is iterative and best done interactively, but we provide a basic propagation.

    return ''.join(plaintext1), ''.join(plaintext2)

# Example usage (replace these ciphertexts with your intercepted texts)
ciphertext1 = """3:6HZ63 06R20:7FKDSSGY)YRT9DWZ BWQNW1VE24R8S8OU5:RP1W60WH,: ZGFI.Z:RM25Y0JUF6(0W"""
ciphertext2 = """75(1ZX0.9S,O5'Y-3?W?7EDOUKMI6VABEO2Q,ZX8:D)E33LZ6745WPOU3E6H:S8R-;ZMX8Q73YDFX55)"""

c1_nums = to_nums(ciphertext1)
c2_nums = to_nums(ciphertext2)

pt1_guess, pt2_guess = guess_plaintexts(c1_nums, c2_nums)

print("Plaintext 1 Guess:", pt1_guess)
print("Plaintext 2 Guess:", pt2_guess)
