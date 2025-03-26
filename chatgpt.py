import collections

alphabet = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.?,-:;'()"
alpha_dict = {char: idx for idx, char in enumerate(alphabet)}
inv_alpha_dict = {idx: char for idx, char in enumerate(alphabet)}

def to_nums(text):
    return [alpha_dict[c] for c in text]

def to_text(nums):
    return ''.join(inv_alpha_dict[n] for n in nums)

def mod46_diff(c1, c2):
    return [(a - b) % 46 for a, b in zip(c1, c2)]

def guess_plaintexts(c1, c2):
    diff = mod46_diff(c1, c2)
    n = len(diff)

    # Hypothesis: assume positions with difference = 0 correspond to same plaintext character
    plaintext1 = ['?'] * n
    plaintext2 = ['?'] * n

    # Using heuristic: assume space (index 0) at positions where diff is common
    for i in range(n):
        if diff[i] == 0:
            plaintext1[i] = plaintext2[i] = '?'
        else:
            # If we guess plaintext1[i] is space (0), plaintext2[i] would be diff[i] mod 46
            # vice versa for plaintext2[i]
            # Check both hypotheses:
            p1_space_p2_char = inv_alpha_dict[diff[i]]
            p2_space_p1_char = inv_alpha_dict[-diff[i] % 46]

            # Heuristic: prefer letters or common punctuation
            if p1_space_p2_char == ' ':
                plaintext1[i] = ' '
                plaintext2[i] = ' '
            elif p2_space_p1_char == ' ':
                plaintext1[i] = ' '
                plaintext2[i] = ' '
            elif p1_space_p2_char.isalpha() and not p2_space_p1_char.isalpha():
                plaintext1[i] = ' '
                plaintext2[i] = p1_space_p2_char
            elif p2_space_p1_char.isalpha() and not p1_space_p2_char.isalpha():
                plaintext2[i] = ' '
                plaintext1[i] = p2_space_p1_char
            else:
                # uncertain, just pick most likely letters
                plaintext1[i] = p2_space_p1_char.lower()
                plaintext2[i] = p1_space_p2_char.lower()

    return ''.join(plaintext1), ''.join(plaintext2)

# Example usage (replace these ciphertexts with your intercepted texts)
ciphertext1 = """3:6HZ63 06R20:7FKDSSGY)YRT9DWZ BWQNW1VE24R8S8OU5:RP1W60WH,: ZGFI.Z:RM25Y0JUF6(0W"""
ciphertext2 = """75(1ZX0.9S,O5'Y-3?W?7EDOUKMI6VABEO2Q,ZX8:D)E33LZ6745WPOU3E6H:S8R-;ZMX8Q73YDFX55)"""

c1_nums = to_nums(ciphertext1)
c2_nums = to_nums(ciphertext2)

pt1_guess, pt2_guess = guess_plaintexts(c1_nums, c2_nums)

print("Plaintext 1 Guess:", pt1_guess)
print("Plaintext 2 Guess:", pt2_guess)
