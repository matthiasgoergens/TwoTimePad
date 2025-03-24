import os
import random

import numpy as np
import tensorflow as tf

import beam
from beam import window_size
from process_corpus import alpha

beam_width = 1000

snippet_length = 100

def load_random_snippet(snippet_length=snippet_length, filename=beam.corpus_filename):
    # Get the total file size
    file_size = os.path.getsize(filename)

    # Check if file is large enough
    if file_size < snippet_length:
        raise ValueError(
            f"File size ({file_size} bytes) is smaller than requested snippet length ({snippet_length} bytes)"
        )

    # Calculate maximum valid starting position
    max_start_pos = file_size - snippet_length

    # Generate random starting position
    start_pos = random.randint(0, max_start_pos)

    # Open the file, seek to the position, and read the snippet
    with open(filename, "rb") as f:
        f.seek(start_pos)
        snippet = f.read(snippet_length)

    return snippet


def diff_plains(snippet_A, snippet_B):
    # Ensure snippets are the same length
    if len(snippet_A) != len(snippet_B):
        raise ValueError(
            f"Snippets must be the same length. Got {len(snippet_A)} and {len(snippet_B)}"
        )

    # Calculate (A - B) % 46 for each corresponding pair of bytes
    differences = [(a - b) % len(alpha) for a, b in zip(snippet_A, snippet_B)]

    # Return as bytes if needed
    # return bytes(differences)

    # Or return as a list of integers
    return differences


def to_text(snippet):
    return "".join([alpha[c] for c in snippet])


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


class ReconstructionState:
    """State representing partial reconstruction of texts A and B"""

    def __init__(self, text_A, text_B, loss_A, loss_B):
        self.text_A = text_A  # List of character indices [0-45]
        self.text_B = text_B  # List of character indices [0-45]
        self.loss_A = loss_A  # Accumulated loss for text A
        self.loss_B = loss_B  # Accumulated loss for text B

    @property
    def total_loss(self):
        return self.loss_A + self.loss_B

    @property
    def length(self):
        return len(self.text_A)

    def context_A(self, window_size=30):
        return self.text_A[-window_size:]

    def context_B(self, window_size=30):
        return self.text_B[-window_size:]


def beam_search(
    differences, get_next_char_losses, beam_width=beam_width, context_size=window_size
):
    """
    Reconstruct texts A and B using beam search.

    Args:
        differences: List of (A-B) % 46 for each position
        get_next_char_losses: Function that returns loss for all characters given a context
        beam_width: Maximum number of states to keep in the beam
        context_size: Size of context window for the language model

    Returns:
        Best reconstructed state (texts A and B with their losses)
    """
    # Initialize with context window filled with spaces (index 0)
    initial_text_A = [0] * context_size
    initial_text_B = [0] * context_size

    # Create initial state
    initial_state = ReconstructionState(
        text_A=initial_text_A, text_B=initial_text_B, loss_A=0.0, loss_B=0.0
    )

    # Initialize beam with the initial state
    beam = [initial_state]

    # Target length to reconstruct
    target_length = context_size + len(differences)

    for pos in range(context_size, target_length):
        new_beam = []
        # # Track best states by context key (Viterbi-like optimization)
        # best_states = {}

        for state in beam:
            # Get the context windows
            context_A = state.context_A(context_size)
            context_B = state.context_B(context_size)

            # Get losses for all possible next characters
            losses_A = get_next_char_losses(context_A)
            losses_B = get_next_char_losses(context_B)

            # Calculate the difference index
            diff_idx = pos - context_size

            # Try all possible next characters for A
            for char_A in range(46):
                # Determine the corresponding character for B
                char_B = (char_A - differences[diff_idx]) % 46

                # Calculate losses
                new_loss_A = state.loss_A + losses_A[char_A]
                new_loss_B = state.loss_B + losses_B[char_B]

                # Create new state
                new_state = ReconstructionState(
                    text_A=state.text_A + [char_A],
                    text_B=state.text_B + [char_B],
                    loss_A=new_loss_A,
                    loss_B=new_loss_B,
                )
                new_beam.append(new_state)

                # Use context as key for Viterbi-like optimization
                # context_key = (
                #     tuple(new_state.context_A()),
                #     tuple(new_state.context_B()),
                # )

                # # Keep only the best state for each context
                # if (
                #     context_key not in best_states
                #     or new_state.total_loss < best_states[context_key].total_loss
                # ):
                #     best_states[context_key] = new_state

        # Select top beam_width states for the next iteration
        beam = sorted(new_beam, key=lambda state: state.total_loss)[
            :beam_width
        ]

        # Print progress every 100 positions
        if pos % 1 == 0:
            print(
                f"Position: {pos}/{target_length}, Best loss: {beam[0].total_loss:.2f}"
            )
            outputA = to_text(beam[0].text_A)
            outputB = to_text(beam[0].text_B)
            print(f"Best A so far: {outputA}")
            print(f"Best B so far: {outputB}")
            diff_so_far = to_text(diff_plains(beam[0].text_A, beam[0].text_B))
            print(f"Best D so far: {diff_so_far}")


    # Return the best state
    return beam[0]


def indices_to_text(indices, charset=" abcdefghijklmnopqrstuvwxyz0123456789.?,-:;'()"):
    """Convert a list of character indices to a string."""
    return "".join(charset[idx] for idx in indices)


def main():
    # TODO: Update this, as we get newer models.
    path = "checkpoints/gru_bn/my_model_epoch_01_batch_40000.keras"
    model = tf.keras.models.load_model(path)

    # Optionally, print the summary to verify.
    model.summary()

    # seed = [0] * window_size  # Replace with your actual seed.
    # probs = predict_next_probabilities(model, seed)
    # print(type(probs))
    # print(probs.sum())
    # print("Next token probabilities:", probs)

    get_next_char_losses = lambda context: predict_next_probabilities(model, context)
    (a, b, diff) = prep()
    best_state = beam_search(diff, get_next_char_losses)
    print("starting reconstruction")
    text_A = indices_to_text(best_state.text_A[30:])  # Skip the initial padding
    text_B = indices_to_text(best_state.text_B[30:])

    print(f"Reconstructed A: {text_A}")
    print(f"Reconstructed B: {text_B}")
    print(f"Total loss: {best_state.total_loss:.2f}")


if __name__ == "__main__":
    main()
    # prep()
