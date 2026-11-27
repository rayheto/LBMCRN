# Streaming interface status

The network uses causal convolutions and temporal recurrent state. The supplied file API retains history across each complete file.

A bounded-memory frame API needs convolution caches, temporal GRU states, analysis history and synthesis overlap-add buffers. Calling the file API independently on adjacent chunks resets history and changes the output.

The frame API and selected-checkpoint device implementation await validation.
