# Implementation notes

## Key width

Currently, both tables take the key width by constructor argument. It could
also be passed via template parameter instead, inlining much of the hash
computations. This was the original design, and incidental measurements seemed
to suggest that this indeed increases performance slightly, at the cost of
having to specify key width at compile time. So this is worth considering in
situations where the key width is known in advance. (The original design can be
simulated by working with custom permutations.)
