import os
import random

import numpy as np
import tensorflow as tf

import beam
from beam import window_size
from process_corpus import alpha

beam_width = 100

def load_random_snippet(snippet_length=1000, filename=beam.corpus_filename):
    # Get the total file size
    file_size = os.path.getsize(filename)
    
    # Check if file is large enough
    if file_size < snippet_length:
        raise ValueError(f"File size ({file_size} bytes) is smaller than requested snippet length ({snippet_length} bytes)")
    
    # Calculate maximum valid starting position
    max_start_pos = file_size - snippet_length
    
    # Generate random starting position
    start_pos = random.randint(0, max_start_pos)
    
    # Open the file, seek to the position, and read the snippet
    with open(filename, 'rb') as f:
        f.seek(start_pos)
        snippet = f.read(snippet_length)
    
    return snippet

def diff_plains(snippet_A, snippet_B):
    # Ensure snippets are the same length
    if len(snippet_A) != len(snippet_B):
        raise ValueError(f"Snippets must be the same length. Got {len(snippet_A)} and {len(snippet_B)}")
    
    # Calculate (A - B) % 46 for each corresponding pair of bytes
    differences = [(a - b) % len(alpha) for a, b in zip(snippet_A, snippet_B)]
    
    # Return as bytes if needed
    # return bytes(differences)
    
    # Or return as a list of integers
    return differences

def to_text(snippet):
    return ''.join([alpha[c] for c in snippet])

def prep():
    textA = load_random_snippet()
    textB = load_random_snippet()
    cipher = diff_plains(textA, textB)
    print(to_text(textA))
    print(to_text(textB))
    print(to_text(cipher))
    return (textA, textB, cipher)


def predict_next_probabilities(model, seed):
    """
    Given a seed (list or array of token indices of length window_size),
    returns the probability distribution over the next token.
    """

    # Ensure the seed is exactly window_size tokens.
    input_seq = np.array(seed[-window_size:]).reshape(1, window_size)

    # Predict raw logits from the model (shape: [1, num_classes]).
    logits = model.predict(input_seq, verbose=0)[0]

    # Compute the log-sum-exp of the logits.
    # log_sum_exp = tf.math.reduce_logsumexp(logits)
    log_probs = tf.nn.log_softmax(logits)

    # Compute the loss in bits for each token: (log(sum(exp(logits))) - logits) / log(2)
    # We ignore the 1/log(2), because our application doesn't care about constant factors.
    return -log_probs.numpy()


# # Example usage:
# # Assume 'alpha' is your list/string of tokens and window_size is defined.
# # For demonstration, create a dummy seed:
# seed = [0] * window_size  # Replace with your actual seed.
# probs = predict_next_probabilities(model, seed)
# print("Next token probabilities:", probs)


def main():
    # TODO: Update this, as we get newer models.
    path = "checkpoints/gru/my_model_epoch_01_batch_230000.keras"
    model = tf.keras.models.load_model(path)

    # Optionally, print the summary to verify.
    model.summary()

    seed = [0] * window_size  # Replace with your actual seed.
    probs = predict_next_probabilities(model, seed)
    print(type(probs))
    print(probs.sum())
    print("Next token probabilities:", probs)


if __name__ == "__main__":
    # main()
    prep()
