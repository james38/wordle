import json
import math
import random
import string

import numpy as np

import gym
from gym import spaces


class WordleEnv(gym.Env):

    def __init__(
        self,
        secret_word=None,
        valid_words_loc='./valids.json',
        solution_words_loc='./words.json',
        n_letters=5,
        max_attempts=6,
        L=4,
        min_guesses=True,
    ):
        self.seed()

        self.load_words(valid_words_loc)
        # construct the possible actions as integer-indexed valid words
        self.action_words = {i: w for i, w in enumerate(self.words)}

        assert isinstance(n_letters, int), "number of letters must be int"
        assert isinstance(max_attempts, int), "max attempts must be int"
        self.n_letters = n_letters
        self.max_attempts = max_attempts

        self.load_solutions(solution_words_loc)
        self.L = L
        if min_guesses:
            self.gen_reward()
        else:
            self.gen_binary_reward()
        self.reward_range = (0,1) # optional to specify

        alphabet = list(string.ascii_lowercase)
        self.len_alphabet = len(alphabet)
        self.alphabet_order = {letter: i for i, letter in enumerate(alphabet)}

        # first there is a discrete input for how many turns have elapsed,
        #  then each letter is represented by 7 inputs
        # of the 7, there are 5 positional inputs per letter, where the
        #  value 2 represents green, 1 as possible yellow, and 0 otherwise
        # the last 2 inputs per letter are metadata, one bit and one discrete
        #  if the first bit is 1, then it has been guessed a number of times
        #   that exceed the number of times it is present in the solution,
        #   (eg. this can include letters that are yellow or green)
        #  the second discrete input is the maximum number of yellows seen
        #   for any guess for that letter (up to 2 for a 5 letter guess)
        #   less any subsequent greens of that letter
        # this is a required variable for the environment
        self.observation_space = spaces.Tuple(
            (
                spaces.MultiDiscrete(
                    [self.max_attempts]
                    + self.len_alphabet * [3] * self.n_letters
                ),
                spaces.MultiBinary(self.len_alphabet),
                spaces.MultiDiscrete(
                    self.len_alphabet * [1 + int(self.n_letters / 2)]
                ),
            )
        )
        self.base_obs = np.zeros(
            1
            + self.len_alphabet * self.n_letters
            + 2 * self.len_alphabet
        ).astype(np.int8)

        # required variable for environment
        self.action_space = spaces.Discrete(len(self.words))

        if secret_word is None:
            self.reset()
        else:
            assert isinstance(secret_word, str), "secret word must be str"
            assert len(secret_word) == n_letters, "secret word wrong length"
            self.reset(secret_word)


    def load_words(self, valid_words_loc):
        with open(valid_words_loc) as f:
            data = f.read()
        self.words = json.loads(data)
        return None


    def load_solutions(self, solution_words_loc):
        with open(solution_words_loc) as f:
            words = json.load(f)
        random.shuffle(words['solutions'])
        self.poss_solutions = words['solutions']
        return None


    def get_random_secret_word(self):
        return random.choice(self.poss_solutions)


    def reset(self, secret_word=None):
        """
        restarts the environment for a new episode
        returns the starting state as a value of self.observation_space
        """
        if secret_word is None:
            self.secret_word = self.get_random_secret_word()
        else:
            self.secret_word = secret_word
        self.correct_chars = set(list(self.secret_word))

        self.n_guesses = 0
        self.overused_letters = list()
        self.valid_words = self.action_words.copy()

        self.state = self.base_obs.copy()
        self.reward = 0
        self.done = False
        self.info = {
            'valid': True,
            'valid_words': self.valid_words,
        }

        return self.state


    def step(self, action):
        """
        takes as input action within self.action_space
        the environment responds with the resultant state
        a 4 tuple of:
          state - same type as reset method return
          reward - a return value received from transitioning to the new state
          done - a boolean for whether the episode is over
          info - an optional dictionary that may be useful to populate
        """
        assert self.action_space.contains(action)
        guess = self.words[action]

        # get current status of individual letters of the guess
        guess_chars = set(list(guess))
        char_pos_vals = {
            j: self.state[
                (1 + self.n_letters * self.alphabet_order[j]):
                (1 + self.n_letters * (self.alphabet_order[j] + 1))
            ]
            for j in guess_chars
        }
        # get current positions' status
        pos_vals = {
            i: self.state[(1 + i):(-2 * self.len_alphabet):self.n_letters]
            for i in range(self.n_letters)
        }
        # get current individual letter status of the secret word
        correct_char_vals = {
            j: self.state[
                (1 + self.n_letters * self.alphabet_order[j]):
                (1 + self.n_letters * (self.alphabet_order[j] + 1))
            ]
            for j in self.correct_chars
        }

        # hard mode constraints
        if self.n_guesses > 0: # none of these constraints apply to 1st guess
            for i,j in enumerate(guess):
                # constraint 1: using known overused letter
                if j in self.overused_letters: # overused cannot contain yellows
                    if char_pos_vals[j][i] != 2:
                        self.info['valid'] = False
                        self.valid_words.pop(guess, None)
                        self.info['valid_words'] = self.valid_words
                        return [self.state, self.reward, self.done, self.info]

                # constraint 2: not using known correct letter
                if np.max(pos_vals[i]) == 2:
                    if j != self.secret_word[i]:
                        self.info['valid'] = False
                        self.valid_words.pop(guess, None)
                        self.info['valid_words'] = self.valid_words
                        return [self.state, self.reward, self.done, self.info]

            # constraint 3: not using known yellow letter
            #  note: it is allowable to guess a yellow again in the same place
            for correct_char, correct_pos_val in correct_char_vals.items():
                if np.any(np.equal(1, correct_pos_val)):
                    # constraint 2 handles that greens are in correct place
                    if guess.count(correct_char) < (
                        1 + np.sum(np.equal(2, correct_pos_val))
                    ):
                        self.info['valid'] = False
                        self.valid_words.pop(guess, None)
                        self.info['valid_words'] = self.valid_words
                        return [self.state, self.reward, self.done, self.info]

        # if the guess passes all the hard mode constraints, then it's valid
        self.n_guesses += 1
        self.state[0] = self.n_guesses

        if guess == self.secret_word:
            self.reward = self.rewards[self.n_guesses - 1]
            self.done = True
        elif self.n_guesses == self.max_attempts:
            self.done = True

        # get char metadata
        # char_meta = {
        #     j: (
        #         # is exceeded
        #         self.state[(self.alphabet_order[j] - 2*self.len_alphabet)],
        #         # max yellows
        #         self.state[(self.alphabet_order[j] - self.len_alphabet)],
        #     ) for j in guess_chars
        # }

        # process the new state after the guess
        # process correct guesses, which should be done first,
        #  not processing yellows and others at the same time
        for i,j in enumerate(guess):
            if j == self.secret_word[i]:
                # set all chars in that position to 0
                self.state[(1 + i):(-2 * self.len_alphabet):self.n_letters] = 0
                # set the correct char in that position to 2
                self.state[1 + self.alphabet_order[j] * self.n_letters + i] = 2
                # if that char had seen yellow(s), increment the yellows down
                if self.state[self.alphabet_order[j] - self.len_alphabet] == 1:
                    self.state[self.alphabet_order[j] - self.len_alphabet] = 0
                    # if that was the only yellow, then remove the other
                    #  positions of that char from being known possible yellow
                    char_pos_vals[j] = np.where(
                        char_pos_vals[j] == 1, 0, char_pos_vals[j]
                    )
                    self.state[
                        (1 + self.n_letters * self.alphabet_order[j]):
                        (1 + self.n_letters * (self.alphabet_order[j] + 1))
                    ] = char_pos_vals[j]
                if self.state[self.alphabet_order[j] - self.len_alphabet] == 2:
                    self.state[self.alphabet_order[j] - self.len_alphabet] = 1

        # get current status of individual letters of the guess
        char_pos_vals = {
            j: self.state[
                (1 + self.n_letters * self.alphabet_order[j]):
                (1 + self.n_letters * (self.alphabet_order[j] + 1))
            ]
            for j in guess_chars
        }

        # process yellows and others
        for i,j in enumerate(guess):
            if (j in self.secret_word) and (j != self.secret_word[i]):
                n_greens = np.sum(np.equal(2, char_pos_vals[j]))
                extra_char = self.secret_word.count(j) - n_greens
                if extra_char > 0:
                    extra_guess_char = guess.count(j) - n_greens
                    n_yellows = min(extra_char, extra_guess_char)
                    # if extra_guess_char is greater than n_yellows,
                    #  it really doesn't matter which is shown as yellow or not,
                    #  as in the game, since we already store the info of how
                    #  many maximum yellows we have, and either position is
                    #  known not to be the correct position, since it's yellow
                    self.state[
                        (self.alphabet_order[j] - self.len_alphabet)
                    ] = max(
                        self.state[self.alphabet_order[j] - self.len_alphabet],
                        n_yellows
                    )

                    # if there are no known possible yellows of that char,
                    #  then set all non-green to possible yellow
                    if not np.any(np.equal(1, char_pos_vals[j])):
                        char_pos_vals[j] = np.where(
                            char_pos_vals[j] == 0, 1, char_pos_vals[j]
                        )
                    else:
                        pass # it was already yellow
                    # set the current position to other, since if it was
                    #  the correct position, then it would have been green
                    char_pos_vals[j][i] = 0
                    self.state[
                        (1 + self.n_letters * self.alphabet_order[j]):
                        (1 + self.n_letters * (self.alphabet_order[j] + 1))
                    ] = char_pos_vals[j]
                else:
                    self.overused_letters.append(j)
                    self.state[
                        (self.alphabet_order[j] - 2*self.len_alphabet)
                    ] = 1
            elif j not in self.secret_word:
                self.overused_letters.append(j)
                self.state[(self.alphabet_order[j] - 2*self.len_alphabet)] = 1
            else:
                pass # the guess was correct and processed already

        # get current positions' status
        pos_vals = {
            i: self.state[(1 + i):(-2 * self.len_alphabet):self.n_letters]
            for i in range(self.n_letters)
        }
        # get current individual letter status of the secret word
        correct_char_vals = {
            j: self.state[
                (1 + self.n_letters * self.alphabet_order[j]):
                (1 + self.n_letters * (self.alphabet_order[j] + 1))
            ]
            for j in self.correct_chars
        }

        self.info['valid'] = True
        # prune a list of valid words for use with random selection agent
        for i, pos_val in pos_vals.items():
            to_pop = list()
            # constraint 1: must use known correct letter in correct place
            if np.max(pos_val) == 2:
                l = list(string.ascii_lowercase)[np.argwhere(pos_val == 2)[0,0]]
                for action, word in self.valid_words.items():
                    if word[i] != l:
                        to_pop.append(action)
                for action in to_pop:
                    self.valid_words.pop(action, None)
            else:
                # constraint 2: must not use known incorrect letter
                for j in self.overused_letters:
                    for action, word in self.valid_words.items():
                        if word[i] == j:
                            to_pop.append(action)
                    for action in to_pop:
                        self.valid_words.pop(action, None)
        # constraint 3: not using known yellow letter
        for correct_char, correct_pos_val in correct_char_vals.items():
            to_pop = list()
            if np.any(np.equal(1, correct_pos_val)):
                for action, word in self.valid_words.items():
                    if word.count(correct_char) < (
                        1 + np.sum(np.equal(2, correct_pos_val))
                    ):
                        to_pop.append(action)
                for action in to_pop:
                    self.valid_words.pop(action, None)
        self.info['valid_words'] = self.valid_words

        return [self.state, self.reward, self.done, self.info]


    def seed(self, seed=None):
        random.seed(seed)
        return None


    def gt_n_poisson(self, L, n):
        """
        returns the inverse of the Poisson CDF with mean, L, at x=n integer
        """
        assert isinstance(n, int), "sample value must be integer"
        sum_prob = 0
        for i in range(n):
            sum_prob += (L**i) / math.factorial(i)
        sum_prob *= np.exp(-L)
        return 1 - sum_prob


    def gen_reward(self):
        """
        this reward structure provides return in proportion to
          the number of guesses for success
        """
        self.rewards = [
            (
                self.gt_n_poisson(self.L, i)
                * (
                    (1 / self.gt_n_poisson(self.L, 1))
                    * (1 - (1 / len(self.poss_solutions)))
                )
            ) for i in range(1, 1 + self.max_attempts)
        ]
        return None


    def gen_binary_reward(self):
        """
        1 for the reward if the word is guessed in at most max_attempts
        """
        self.rewards = [1 for i in range(1, 1 + self.max_attempts)]
        return None


    def gen_possible_curr_board(self):
        """
        *unused idea*
        calculate total possible board states
        [65, 80, 50, 20, 5]
          for (0, 1, 2, 3, 4) correct placings
          and all possible numbers of correct letters incorrectly placed
           including no correct letters incorrectly placed
        """
        self.board = [
            (
                math.comb(self.n_letters, n)
                * sum(
                    [
                        math.perm(self.n_letters - n - 1, j)
                        for j in range(self.n_letters - n)
                    ]
                )
            ) for n in range(self.n_letters)
        ]
        return None


def main():
    wordle = WordleEnv()
    return None


if __name__ == "__main__":
    main()
