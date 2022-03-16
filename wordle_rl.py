import json
import math
import random
import string

import numpy as np
import pandas as pd

import gym
from gym import spaces, utils, wrappers

import torch
import torch.nn as nn


class WordleEnv(gym.Env):

    def __init__(
        self,
        secret_word=None,
        valid_words_loc='./valids.json',
        solution_words_loc='./words.json',
        n_letters=5,
        max_attempts=6,
        L=3.4,
    ):
        self.seed()

        self.load_words(valid_words_loc)
        # construct the possible actions as integer-indexed valid words
        self.action_words = [(i, w) for i, w in enumerate(self.words)]

        assert type(n_letters) is int, "number of letters must be int"
        assert type(max_attempts) is int, "number of max attempts must be int"
        self.n_letters = n_letters
        self.max_attempts = max_attempts
        self.L = L
        self.gen_reward()
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
        )

        # required variable for environment
        self.action_space = spaces.Discrete(len(self.words))

        self.load_solutions(solution_words_loc)
        if secret_word is None:
            self.reset()
        else:
            assert type(secret_word) is str, "secret word must be str"
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
        self.non_letters = {
            char: i for char, i in self.alphabet_order.items()
            if char not in self.secret_word
        }

        self.state = self.base_obs.copy()
        self.reward = 0
        self.done = False
        self.info = dict()

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
                        return [self.state, self.reward, self.done, self.info]

                # constraint 2: not using known correct letter
                if np.max(pos_vals[i]) == 2:
                    if j != self.secret_word[i]:
                        return [self.state, self.reward, self.done, self.info]

            # constraint 3: not using known yellow letter
            for correct_char, correct_pos_val in correct_char_vals.items():
                if np.any(np.equal(1, correct_pos_val)):
                    # constraint 2 handles that greens are in correct place
                    if guess.count(correct_char) < (
                        1 + np.sum(np.equal(2, correct_pos_val))
                    ):
                        return [self.state, self.reward, self.done, self.info]

        # if the guess passes all the hard mode constraints, then it's valid
        self.n_guesses += 1
        self.state[0] = self.n_guesses

        if guess == self.secret_word:
            self.reward = self.rewards[self.n_guesses - 1]
            self.done = True

        if self.n_guesses == self.max_attempts:
            self.done = True

        # get char metadata
        char_meta = {
            j: (
                # is exceeded
                self.state[2*(self.alphabet_order[j] - self.len_alphabet)],
                # max yellows
                self.state[(self.alphabet_order[j] - self.len_alphabet)],
            ) for j in guess_chars
        }

        # vals = np.zeros(self.n_letters)
        # for i,j in enumerate(guess):
        #     vals[i] = self.state[
        #         (
        #             1
        #             + (self.alphabet_order[j] * self.n_letters)
        #             + i
        #         )
        #     ]
        # n_greens = np.sum(vals == 2)
        # n_yellows = np.sum(vals == 1)

        # for i,j in enumerate(guess):
        #     char_pos_vals[j]
        #     char_meta[j]
        #     pos_vals[i]

        # process the new state after the guess
        # process correct guesses, which should be done first,
        #  not processing yellows and others at the same time
        for i,j in enumerate(guess):
            if j == self.secret_word[i]:
                # set all other chars in that position to 0
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
        for i,j in enumerate(guess):
            if (j in self.secret_word) and (j != self.secret_word[i]):
                if (
                    self.secret_word.count(j)
                    > np.sum(np.equal(2, char_pos_vals[j]))
                ):
                    if not np.any(np.equal(1, char_pos_vals[j])):
                        # this should work
                        char_pos_vals[j] = np.where(
                            char_pos_vals[j] == 0, 1, char_pos_vals[j]
                        )
                    else:
                        pass # it was already yellow
                    char_pos_vals[j][i] = 0
                    self.state[
                        (1 + self.n_letters * self.alphabet_order[j]):
                        (1 + self.n_letters * (self.alphabet_order[j] + 1))
                    ] = char_pos_vals[j]
                else:
                    self.overused_letters.append(j)
            elif j not in self.secret_word:
                self.overused_letters.append(j)


        return [self.state, self.reward, self.done, self.info]


    def gt_n_poisson(self, L, n):
        """
        returns the inverse of the Poisson CDF with mean, L, at x=n integer
        """
        assert type(n) is int, "sample value must be integer"
        sum_prob = 0
        for i in range(n):
            sum_prob += (L**i) / math.factorial(i)
        sum_prob *= np.exp(-L)
        return 1 - sum_prob


    def gen_reward(self):
        """
        returns the reward structure that provides return in proportion to
        the number of guesses for success
        """
        self.rewards = [
            self.gt_n_poisson(self.L, i)
            for i in range(1, 1 + self.max_attempts)
        ]
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
