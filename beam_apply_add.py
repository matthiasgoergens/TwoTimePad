import os
import random

import numpy as np
import tensorflow as tf
import tempfile

import beam
from beam import window_size
from process_corpus import alpha

beam_width = 2_000

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


def add_plains(snippet_A, snippet_B):
    # Ensure snippets are the same length
    if len(snippet_A) != len(snippet_B):
        raise ValueError(
            f"Snippets must be the same length. Got {len(snippet_A)} and {len(snippet_B)}"
        )

    # Calculate (A + B) % 46 for each corresponding pair of bytes
    differences = [(a + b) % len(alpha) for a, b in zip(snippet_A, snippet_B)]

    # Return as bytes if needed
    # return bytes(differences)

    # Or return as a list of integers
    return differences


def to_text(snippet):
    # print(snippet)
    # print(alpha)
    # bb = [alpha[c] for c in snippet][:100]
    # print(bb)
    # print(bytes(bb))
    return bytes([alpha[c] for c in snippet])


def prep():
    # This is a hack, something is wrong with my conversion, and I lose a few bytes.
    # I hope those aren't at the beginning.
    trunc = 8000
    textA = bytes(open("examples/unsong.bytes", "rb").read())
    # textB = load_random_snippet()
    textB = bytes(open("examples/worm.bytes", "rb").read())
    trunc = min(len(textA), len(textB))
    textA = textA[:trunc]
    textB = textB[:trunc]

    cipher = add_plains(textA, textB)
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


def predict_next_probabilities_batched(model, seeds):
    """
    seeds: A numpy array of shape (batch_size, window_size)
    returns: A numpy array of shape (batch_size, window_size, len(alpha)), representing
             negative log-probabilities for each timestep and token.
    """
    # Predict logits for all inputs at once (shape: [batch_size, window_size, num_classes])
    logits = model.predict(seeds, verbose=0)

    # Compute log softmax (log-probabilities)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    # Convert to negative log probabilities
    neg_log_probs = -log_probs.numpy()  # shape: (batch_size, window_size, len(alpha))
    return neg_log_probs


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

    def context_A(self, window_size=window_size):
        return self.text_A[-window_size:]

    def context_B(self, window_size=window_size):
        return self.text_B[-window_size:]


def beam_search(differences, model, beam_width=beam_width, context_size=window_size):
    initial_text_A = [0] * context_size
    initial_text_B = [0] * context_size

    initial_state = ReconstructionState(
        text_A=initial_text_A, text_B=initial_text_B, loss_A=0.0, loss_B=0.0
    )

    beam = [initial_state]
    target_length = context_size + len(differences)

    for pos in range(context_size, target_length):
        diff_idx = pos - context_size
        candidates = []

        # Prepare batch inputs for A and B
        contexts_A = np.array([state.context_A(context_size) for state in beam])
        contexts_B = np.array([state.context_B(context_size) for state in beam])

        # Batch prediction
        losses_A = predict_next_probabilities_batched(model, contexts_A)
        losses_B = predict_next_probabilities_batched(model, contexts_B)

        # Only take the last timestep's predictions
        losses_A = losses_A[:, -1, :]  # shape (beam_width, len(alpha))
        losses_B = losses_B[:, -1, :]

        # d = A - B
        # B = A - d

        # s = A + B
        # B = s - A
        # Expand beam states
        for i, state in enumerate(beam):
            for char_A in range(46):
                char_B = (differences[diff_idx] - char_A) % 46
                new_loss_A = state.loss_A + losses_A[i, char_A]
                new_loss_B = state.loss_B + losses_B[i, char_B]

                new_state = ReconstructionState(
                    text_A=state.text_A + [char_A],
                    text_B=state.text_B + [char_B],
                    loss_A=new_loss_A,
                    loss_B=new_loss_B,
                )
                candidates.append(new_state)

        # Keep top beam_width states
        beam = sorted(candidates, key=lambda state: state.total_loss)[:beam_width]

        def present_beam1(beam1):
            return to_text(beam1[context_size:])

        best = beam[0].total_loss
        worst = beam[-1].total_loss
        print(
            f"Position: {pos}/{target_length}, Best loss: {best:.2f}, Worst loss: {worst:.2f}, Diff: {worst-best:.2f}"
        )

        print(f"Best A so far: {present_beam1(beam[0].text_A)}")
        print(f"Worst A still: {present_beam1(beam[-1].text_A)}")
        print(f"Best B so far: {present_beam1(beam[0].text_B)}")
        print(f"Worst B still: {present_beam1(beam[-1].text_B)}")
        # diff_so_far = to_text(diff_plains(beam[0].text_A, beam[0].text_B))
        # print(f"Best D so far: {diff_so_far}")

    return beam[0]


output_file = "best_prediction.txt"


def indices_to_text(indices, charset=" abcdefghijklmnopqrstuvwxyz0123456789.?,-:;'()"):
    """Convert a list of character indices to a string."""
    return "".join(charset[idx] for idx in indices)


def main():
    # path = "checkpoints/rnn_lstm_final_less_decay/epoch_1345.keras"
    # path = "checkpoints/lstm_1_layer/epoch_1346.keras"
    path = "checkpoints/lstm_ablated_double_layers_0.05dropout_long/epoch_660.keras"
    model = tf.keras.models.load_model(path)
    model.summary()

    (a, b, diff) = prep()

    best_state = beam_search(diff, model)  # Pass model directly now
    print("Reconstruction complete.")
    text_A = indices_to_text(best_state.text_A[window_size:])
    text_B = indices_to_text(best_state.text_B[window_size:])

    print(f"Reconstructed A: {text_A}")
    print(f"Reconstructed B: {text_B}")
    print(f"Total loss: {best_state.total_loss:.2f}")


if __name__ == "__main__":
    main()
    # prep()
