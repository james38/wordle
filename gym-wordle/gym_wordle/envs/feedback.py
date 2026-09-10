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


def hard_mode_mask(green_pos, min_count, W_letters, W_counts):
    """Real Wordle hard mode over the whole word list.

    green_pos (B,n) int8, -1 where no green is known; min_count (B,26) int8.
    Word w is legal for game b iff every known green matches and, for every
    letter, w has at least min_count[b, l] copies. Greys are not forbidden.
    Returns (B,V) bool.
    """
    has_green = green_pos >= 0                                        # (B,n)
    match = W_letters.unsqueeze(0) == green_pos.unsqueeze(1)          # (B,V,n)
    r1 = (match | ~has_green.unsqueeze(1)).all(-1)                    # (B,V)
    r2 = (W_counts.unsqueeze(0) >= min_count.unsqueeze(1)).all(-1)    # (B,V)
    return r1 & r2


def consistent_mask(guess_letters, colours, n_guesses, W_letters, W_counts):
    """Words whose colouring against every past guess reproduces the record.

    guess_letters, colours: (B,T,n) int8; n_guesses (B,) long says how many
    of the T rows are real. Uses the count characterisation of Wordle
    colouring: for a past guess g with colours c, a word w is consistent iff
      1. for each position p, (w[p] == g[p]) == (c[p] == GREEN), and
      2. for each letter l in g, with k = greens+yellows of l in g:
         count_w(l) == k if g has a grey copy of l, else count_w(l) >= k.
    Returns (B,V) bool. Never fed to the policy.
    """
    B, T, n = guess_letters.shape
    dev = guess_letters.device
    V = W_letters.shape[0]
    ok = torch.ones(B, V, dtype=torch.bool, device=dev)
    WC_T = W_counts.t().contiguous()                                  # (26,V) int8
    for t in range(T):
        active = n_guesses > t                                        # (B,)
        if not bool(active.any()):
            break
        g = guess_letters[:, t].long()                                # (B,n)
        c = colours[:, t]                                             # (B,n) int8
        is_green = c == GREEN
        match = W_letters.unsqueeze(0) == g.unsqueeze(1).to(W_letters.dtype)  # (B,V,n)
        r1 = (match == is_green.unsqueeze(1)).all(-1)                 # (B,V)

        onehot = torch.nn.functional.one_hot(g, N_ALPHABET).to(torch.int8)   # (B,n,26)
        need = (onehot * (c != GREY).unsqueeze(-1).to(torch.int8)).sum(1, dtype=torch.int8)    # (B,26)
        has_grey = (onehot * (c == GREY).unsqueeze(-1).to(torch.int8)).sum(1) > 0            # (B,26)
        need_g = need.gather(1, g)                                    # (B,n)
        exact_g = has_grey.gather(1, g)                               # (B,n)
        cnt_g = WC_T[g]                                               # (B,n,V) int8
        ge = cnt_g >= need_g.unsqueeze(-1)
        eq = cnt_g == need_g.unsqueeze(-1)
        r2 = torch.where(exact_g.unsqueeze(-1), eq, ge).all(1)        # (B,V)

        ok &= ~active.unsqueeze(1) | (r1 & r2)
    return ok
