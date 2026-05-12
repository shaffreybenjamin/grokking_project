import numpy as np
import matplotlib.pyplot as plt
from itertools import product

def all_sequences(max_len: int, alphabet_size: int = 3) -> list[np.ndarray]:
    return [
        np.array(seq)
        for length in range(1, max_len + 1)
        for seq in product(range(alphabet_size), repeat=length)
    ]


def zero_one_random(p: float) -> np.array:
    """Creates a transition matrix for the Zero One Random (Z1R) Process.

    Steady-state distribution = [1, 1, 1] / 3
    """
    assert 0 <= p <= 1
    q = 1 - p
    return np.array(
        [
            [
                [0, 1, 0],
                [0, 0, 0],
                [q, 0, 0],
            ],
            [
                [0, 0, 0],
                [0, 0, 1],
                [p, 0, 0],
            ],
        ]
    )

def mess3(x: float, a: float) -> np.ndarray:
    """Creates a transition matrix for the Mess3 Process."""
    b = (1 - a) / 2
    y = 1 - 2 * x

    ay = a * y
    bx = b * x
    by = b * y
    ax = a * x

    return np.array(
        [
            [
                [ay, bx, bx],
                [ax, by, bx],
                [ax, bx, by],
            ],
            [
                [by, ax, bx],
                [bx, ay, bx],
                [bx, ax, by],
            ],
            [
                [by, bx, ax],
                [bx, by, ax],
                [bx, bx, ay],
            ],
        ]
    )


def belief_state(sequence, transition_matrix, initial_vec):
    vect = initial_vec
    for obs in sequence:
        vect = vect @ transition_matrix[obs]
    return vect/np.sum(vect)

def belief_state_update(belief_state, transition_matrix, obs):
    return belief_state @ transition_matrix[obs] / np.sum(belief_state @ transition_matrix[obs])

x = 0.15
a = 0.2
max_len = 6

transition_matrix = mess3(x, a)
initial_vec = np.array([[1/3, 1/3, 1/3]])
sequences = all_sequences(max_len)

beliefs = np.array([belief_state(seq, transition_matrix, initial_vec).flatten() for seq in sequences])

vertices = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]])
points = beliefs @ vertices

fig, ax = plt.subplots(figsize=(7, 6))
triangle = plt.Polygon(vertices, fill=False, edgecolor="black")
ax.add_patch(triangle)
ax.scatter(points[:, 0], points[:, 1], s=8, alpha=0.6)
ax.set_aspect("equal")
ax.axis("off")
plt.show()