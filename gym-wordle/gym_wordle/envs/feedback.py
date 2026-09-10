"""Pure batched Wordle feedback functions. Torch, device-agnostic, stateless.

Every function takes tensors and returns tensors on the same device. None
of them look at the solution list: they see letters, colours and the
valid-word matrices only.
"""
import torch

GREY, YELLOW, GREEN = 0, 1, 2
N_ALPHABET = 26


def colour(guess_letters, secret_letters):
    """Standard Wordle colouring, batched. Shapes broadcast to (..., n).

    Greens first. Then, scanning positions left to right, a non-green cell
    is yellow iff the secret still has an unmatched copy of that letter;
    each yellow consumes one copy. Returns int8 in {GREY, YELLOW, GREEN}.
    """
    guess, secret = torch.broadcast_tensors(guess_letters.long(), secret_letters.long())
    shape = guess.shape
    n = shape[-1]
    guess = guess.reshape(-1, n)
    secret = secret.reshape(-1, n)
    b = guess.shape[0]
    dev = guess.device

    green = guess == secret
    remaining = torch.zeros(b, N_ALPHABET, dtype=torch.int32, device=dev)
    remaining.scatter_add_(1, secret, torch.ones_like(secret, dtype=torch.int32))
    remaining.scatter_add_(1, guess, -green.to(torch.int32))

    out = torch.full((b, n), GREY, dtype=torch.int8, device=dev)
    out[green] = GREEN
    rows = torch.arange(b, device=dev)
    for p in range(n):
        letter = guess[:, p]
        avail = (remaining[rows, letter] > 0) & ~green[:, p]
        out[avail, p] = YELLOW
        remaining[rows, letter] -= avail.to(torch.int32)
    return out.reshape(shape)
