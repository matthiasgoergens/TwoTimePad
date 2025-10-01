def make_model():
    relu = PReLU
    embedded_output_dim = 64

    flatten = Lambda(
        lambda x: tf.reshape(x, [tf.shape(x)[0], window_size * embedded_output_dim])
    )

    inputs = Input(shape=(window_size,))
    # 46 unique tokens mapped to 64-dim embeddings
    outputs = Sequential(
        [
            Embedding(input_dim=46, output_dim=embedded_output_dim),
            flatten,
        ]
    )(inputs)

    # outputs = Flatten()(embedded)
    height = 10
    blowup = 256
    for i in range(height):
        outputs = cat(
            outputs,
            Sequential(
                [
                    BatchNormalization(),
                    relu(),
                    Dense(blowup),
                ]
            )(outputs),
        )
    outputs = Dense(len(alpha))(outputs)
    model = model = Model([inputs], [outputs])
    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    model.summary()
    checkpoint_dir = "checkpoints/cumulative1"
    return (model, checkpoint_dir)


def make_model_gru():
    embedded_output_dim = 46

    model = Sequential(
        [
            Input(shape=(window_size,)),
            Embedding(input_dim=len(alpha), output_dim=embedded_output_dim),
            LayerNormalization(),
            GRU(256, return_sequences=True),
            LayerNormalization(),
            GRU(256, return_sequences=True),
            LayerNormalization(),
            GRU(256),
            LayerNormalization(),
            Dense(len(alpha)),
        ]
    )

    model.compile(
        optimizer=tf.optimizers.Adam(global_clipnorm=1.0),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.summary()
    checkpoint_dir = "checkpoints/gru_ln3_twice"
    return (model, checkpoint_dir)


def make_model_relu_rnn():
    embedded_output_dim = 64

    model = tf.keras.Sequential(
        [
            Input(shape=(window_size,)),
            Embedding(input_dim=len(alpha), output_dim=embedded_output_dim),
            BatchNormalization(),
            # First SimpleRNN layer with ReLU
            SimpleRNN(768, activation="linear", return_sequences=True),
            PReLU(),
            BatchNormalization(),
            # Second SimpleRNN layer with ReLU
            SimpleRNN(768, activation="linear", return_sequences=True),
            PReLU(),
            BatchNormalization(),
            # Third SimpleRNN layer with ReLU
            SimpleRNN(768, activation="linear"),
            PReLU(),
            BatchNormalization(),
            Dense(len(alpha)),
        ]
    )

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.summary()
    checkpoint_dir = "checkpoints/relu_rnn"
    return (model, checkpoint_dir)


def make_model_dense_rnn():
    embedded_output_dim = len(alpha)
    rnn_units = 256
    num_layers = 10  # Easily adjustable number of layers

    # Use Functional API for complex connectivity
    inputs = Input(shape=(window_size,))

    # Embedding layer
    x = Embedding(input_dim=len(alpha), output_dim=embedded_output_dim)(inputs)
    x = BatchNormalization()(x)

    # Store sequence outputs for concatenation
    sequence_outputs = [x]

    # Create RNN layers with dense connectivity
    for i in range(num_layers):
        # Last layer doesn't need to return sequences
        return_sequences = i < num_layers - 1

        # For layers after the first, concatenate all previous sequence outputs
        if i > 0:
            x = Concatenate(axis=2)(sequence_outputs)

        # Apply RNN layer with PReLU and BatchNorm
        x = SimpleRNN(
            rnn_units, activation="linear", return_sequences=return_sequences
        )(x)
        x = PReLU()(x)
        x = BatchNormalization()(x)

        # Add to sequence outputs if it returns sequences
        if return_sequences:
            sequence_outputs.append(x)

    # Final prediction layer
    outputs = Dense(len(alpha))(x)

    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.summary()
    checkpoint_dir = "checkpoints/dense_rnn_for_10"
    return (model, checkpoint_dir)


def make_model_condensed_skip_rnn():
    embedded_output_dim = len(alpha)
    rnn_units = 256
    condensed_dim = 256  # Size of the compressed skip connections
    num_layers = 10

    # Use Functional API
    inputs = Input(shape=(window_size,))

    # Embedding layer
    x = Embedding(input_dim=len(alpha), output_dim=embedded_output_dim)(inputs)
    x = LayerNormalization()(x)

    # Store all sequence outputs for skip connections
    all_outputs = [x]
    current_output = x

    # Create RNN layers with learnable condensed skip connections
    for i in range(num_layers):
        # All layers return sequences now
        return_sequences = True

        # Condense half of previous outputs through a learnable projection
        if i > 0:
            # Concatenate half of previous outputs
            combined = Concatenate(axis=2)(all_outputs[-1::-2])

            # Learnable projection to reduce dimensionality
            skip_projection = TimeDistributed(
                Dense(condensed_dim, activation="linear")
            )(combined)
            skip_projection = LayerNormalization()(skip_projection)
            skip_projection = PReLU()(skip_projection)

            # Feed the condensed representation to the RNN layer
            current_output = SimpleRNN(
                rnn_units,
                activation="linear",
                kernel_initializer=Orthogonal(gain=1.2),
                recurrent_initializer=Orthogonal(gain=1.2),
                return_sequences=return_sequences,
            )(skip_projection)
        else:
            # First layer just processes the embedding
            current_output = SimpleRNN(
                rnn_units,
                activation="linear",
                kernel_initializer=Orthogonal(gain=1.2),
                recurrent_initializer=Orthogonal(gain=1.2),
                return_sequences=return_sequences,
            )(current_output)

        current_output = LayerNormalization()(current_output)
        current_output = PReLU()(current_output)

        # Save this output for future skip connections
        all_outputs.append(current_output)

    # Final prediction layer - predict at each timestep
    outputs = TimeDistributed(Dense(len(alpha)))(Concatenate(axis=2)(all_outputs))

    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.optimizers.Adam(
            global_clipnorm=0.5,
            weight_decay=1e-4,
        ),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.summary()
    checkpoint_dir = "rnn_prelu_skip"
    return (model, checkpoint_dir)
