from __future__ import annotations

import itertools
import math
from abc import ABC, abstractmethod
from typing import List

from PIL import Image
from numpy.fft import *
from numpy.random import rand, randint, random, uniform, choice, normal

from datastructures.functiontree import *
from utils.safefunctions import *


class Representation(ABC):
    """
    A super class
    """
    animate = False  # does the representation move?
    go_back = False  # if the representation is animated, should the gif go back to the first frame on display?

    def __init__(self, params):
        self.params = params
        self.statistics = []

    def get_pixel_range(self, resx, resy):
        if resx < resy:
            return 2, 2 * (resy / resx)
        else:
            return 2 * (resx / resy), 2

    @abstractmethod
    def crossover(self, other) -> List[Representation]:
        pass

    @abstractmethod
    def mutate(self) -> Representation:
        pass

    @abstractmethod
    def evaluate(self):
        """
        For now, we can keep this function empty, since the evaluation happens in the view, and the fitness scores are
        passed to the evolutionaryframework via the control.update_scores function. In another use case, this function
        may contain a computation.
        """
        pass

    @abstractmethod
    def get_output(self, resx, resy) -> np.array:
        """
        :return output: the product to evaluate (in our case, this will be an array of pixel values)
        """
        pass

    def copy(self):
        pass

    def get_statistics(self):
        pass

    def latex_repr(self):
        pass


# TODO Make a more general tree representation to avoid duplicate code
# For example: Tree(1,1,1) could make a tree with 3 separate 1-channel trees, and Tree(3) could make a tree with 1
# 3-channel tree
class NChannelTreeRepresentation(Representation):

    def __str__(self):
        tree_strings = [str(t) for t in self.trees]
        return "\n".join(tree_strings)

    def __init__(self, params):
        super().__init__(params)
        self.name = 'tree'
        self.statistics = 'max_depth', 'n_nodes'
        math.sin(1)
        self.F_bin = [
            (np.add, "+")
            , (np.subtract, "-")
            , (np.multiply, "*")
            , (safediv, "/")
            # , (safepow, "^")
            , (np.minimum, "min")
            , (np.maximum, "max")]
        self.F_un = [
            (np.sin, "sin"),
            (np.cos, "cos"),
            (np.abs, "abs"),
            (safesqrt, "sqrt")]
        self.T = ['x', 'y', 0.618]
        # TODO initialize random constants that persist through crossover and mutation
        # TODO make sampling between terminals uniform
        self.params['tree_params']['n_un'] = len(self.F_un)
        self.params['tree_params']['n_bin'] = len(self.F_bin)
        self.params['tree_params']['n_t'] = len(self.T)
        self.params['tree_params']['branch_samples'] = self.F_un + self.F_bin
        self.params['tree_params']['leaf_samples'] = self.T

        self.trees = []
        for i in range(self.params['n_channels']):
            self.trees.append(self.make_tree(self.params['tree_params']))

    def make_tree(self, params):
        r = np.random.rand(1)
        param_copy = params.copy()

        if (param_copy['method'] == 'hybrid') and (r < 0.5):
            param_copy['method'] = 'full'
        elif (param_copy['method'] == 'hybrid') and (r >= 0.5):
            param_copy['method'] = 'grow'

        if param_copy['method'] == 'full':
            idx = random.randint(low=0, high=len(param_copy['branch_samples']))
            if idx < param_copy['n_un']:
                return UnaryBranch(param_copy, idx)
            else:
                return BinaryBranch(param_copy, idx)

        elif param_copy['method'] == 'grow':
            idx = random.randint(low=0, high=len(param_copy['branch_samples']) + param_copy['n_t'])
            if idx < param_copy['n_un']:
                return UnaryBranch(param_copy, idx)
            elif idx < param_copy['n_un'] + param_copy['n_bin']:
                return BinaryBranch(param_copy, idx)
            else:
                return Leaf(params, idx - len(params['branch_samples']))

    def crossover(self, other: NChannelTreeRepresentation) -> List[NChannelTreeRepresentation]:
        # Make a copy of all trees, receiving copies
        my_tree_copies_to = self.tree_copy()
        other_tree_copies_to = other.tree_copy()

        # Make another copy of all trees, used to copy stuff from
        other_tree_copies_from = other.tree_copy()
        my_tree_copies_from = self.tree_copy()
        # other_tree_copies_to = other.tree_copy()
        # for i in range(self.params['n_channels']):
        #     mycopy_to = my_tree_copies_to[i]
        #     othercopy_from = other_tree_copies_from[i]

        # Select a random color layer
        rand_tree_idx = np.random.randint(0, self.params['n_channels'])
        my_copy_to = my_tree_copies_to[rand_tree_idx]
        other_copy_from = other_tree_copies_from[rand_tree_idx]
        my_copy_from = my_tree_copies_from[rand_tree_idx]
        other_copy_to = other_tree_copies_to[rand_tree_idx]

        # If the trees are big enough, select two random nodes and transplant them
        if my_copy_to.number_of_nodes() > 1 and other_copy_to.number_of_nodes() > 1:
            idx_self = np.random.randint(1, my_copy_to.number_of_nodes())
            idx_other = np.random.randint(1, other_copy_to.number_of_nodes())

            self.transplant(my_copy_to, other_copy_from, idx_self, idx_other)
            self.transplant(other_copy_to, my_copy_from, idx_other, idx_self)

        # Select two random nodes within the tree
        idx_self = np.random.randint(1, my_copy_to.number_of_nodes())
        idx_other = np.random.randint(1, other_copy_to.number_of_nodes())

        # Transplant
        self.transplant(my_copy_to, other_copy_from, idx_self, idx_other)
        self.transplant(other_copy_to, my_copy_from, idx_other, idx_self)

        # Create children
        tree1 = NChannelTreeRepresentation(self.params)
        tree1.trees = my_tree_copies_to
        tree2 = NChannelTreeRepresentation(self.params)
        tree2.trees = other_tree_copies_to
        return [tree1, tree2]

    def transplant(self, target, source, ind_to, ind_from):
        """
        Transplant the node with index ind_from in other to index ind_to in self
        :param target: TreeRepresentation
        :param source: TreeRepresentation
        :param ind_to: int
        :param ind_from: int
        """
        node_to_be_replaced = target.get_nth_node(ind_to)
        replacement_node = source.get_nth_node(ind_from)
        parent_with_new_child = node_to_be_replaced.parent
        parent_with_new_child.children[node_to_be_replaced.key] = replacement_node
        replacement_node.parent = parent_with_new_child

    # TODO Implement Mutation (We are not using it now so it's okay for now)
    def mutate(self):
        pass

    def get_output(self, resx, resy) -> np.array:

        xmax, ymax = self.get_pixel_range(resx, resy)
        xmin = -xmax
        ymin = -ymax
        xspace = np.linspace(xmin, xmax, resx)
        yspace = np.linspace(ymin, ymax, resy)

        xx, yy = np.meshgrid(yspace, xspace)
        outs = np.zeros((self.params['n_channels'], resx, resy))
        for i in range(self.params['n_channels']):
            this_out = np.ones_like(xx) * self.trees[i].evaluate(yy.astype(np.float64), xx.astype(np.float64))
            this_out_max, this_out_min = this_out.max(), this_out.min()
            this_out_normalized = (this_out - this_out_min) / ((this_out_max - this_out_min) + 0.001)
            outs[i] = this_out_normalized
        return outs.swapaxes(0, 2)

    def evaluate(self):
        pass

    def tree_copy(self):
        tree_copies = []
        for t in self.trees:
            tree_copies.append(t.copy(None))
        return tree_copies

    def get_statistics(self):
        maxdepth = np.max(np.array([i.max_depth() for i in self.trees]))
        number_of_nodes = np.sum(np.array([i.number_of_nodes() for i in self.trees]))
        return [maxdepth, number_of_nodes]


class NChannelTreeRepresentationV2(Representation):

    def __str__(self):
        return str(self.tree)

    def latex_repr(self):
        return self.tree.latex_repr()

    def __init__(self, params):
        super().__init__(params)
        self.name = 'tree_v2'
        self.statistics = 'max_depth', 'n_nodes'
        self.F_bin = [
            (np.add, "+")
            , (np.subtract, "-")
            , (np.multiply, "*")
            , (safediv, "/")
            , (safepow, "^")
            # , (np.minimum, "min")
            # , (np.maximum, "max")
        ]
        self.F_un = [
            (np.sin, "sin"),
            (np.cos, "cos"),
            (np.abs, "abs"),
            (safesqrt, "sqrt")]
        self.T = ['x', 'y', 'z', 0.618]
        # TODO initialize random constants that persist through crossover and mutation
        # TODO make sampling between terminals uniform
        self.params['tree_params']['n_un'] = len(self.F_un)
        self.params['tree_params']['n_bin'] = len(self.F_bin)
        self.params['tree_params']['n_t'] = len(self.T)
        self.params['tree_params']['branch_samples'] = self.F_un + self.F_bin
        self.params['tree_params']['leaf_samples'] = self.T
        self.tree = self.make_tree(self.params['tree_params'])

    def make_tree(self, params: dict):
        r = np.random.rand(1)
        param_copy = params.copy()

        if (param_copy['method'] == 'hybrid') and (r < 0.5):
            param_copy['method'] = 'full'
        elif (param_copy['method'] == 'hybrid') and (r >= 0.5):
            param_copy['method'] = 'grow'

        if param_copy['method'] == 'full':
            idx = random.randint(low=0, high=len(param_copy['branch_samples']))
            if idx < param_copy['n_un']:
                return UnaryBranch(param_copy, idx)
            else:
                return BinaryBranch(param_copy, idx)
        elif param_copy['method'] == 'grow':
            idx = random.randint(low=0, high=len(param_copy['branch_samples']) + param_copy['n_t'])
            if idx < param_copy['n_un']:
                return UnaryBranch(param_copy, idx)
            elif idx < param_copy['n_un'] + param_copy['n_bin']:
                return BinaryBranch(param_copy, idx)
            else:
                return Leaf(param_copy, idx - len(param_copy['branch_samples']))

    def crossover(self, other: NChannelTreeRepresentationV2) -> List[NChannelTreeRepresentationV2]:
        mycopy_to = self.tree_copy()
        othercopy_from = other.tree_copy()

        mycopy_from = self.tree_copy()
        othercopy_to = other.tree_copy()

        tree1 = NChannelTreeRepresentationV2(self.params)
        tree2 = NChannelTreeRepresentationV2(self.params)

        # If the trees are big enough, select two random nodes and transplant them
        if mycopy_to.number_of_nodes() > 1 and othercopy_to.number_of_nodes() > 1:
            idx_self = np.random.randint(1, mycopy_to.number_of_nodes())
            idx_other = np.random.randint(1, othercopy_to.number_of_nodes())

            self.transplant(mycopy_to, othercopy_from, idx_self, idx_other)
            self.transplant(othercopy_to, mycopy_from, idx_other, idx_self)

        mycopy_to.refresh_depth()
        othercopy_to.refresh_depth()

        tree1.tree = mycopy_to
        tree2.tree = othercopy_to
        return [tree1, tree2]

    def transplant(self, target, source, ind_to, ind_from):
        """
        Transplant the node with index ind_from in other to index ind_to in self
        :param target: TreeRepresentation
        :param source: TreeRepresentation
        :param ind_to: int
        :param ind_from: int
        """
        node_to_be_replaced = target.get_nth_node(ind_to)
        replacement_node = source.get_nth_node(ind_from)
        parent_with_new_child = node_to_be_replaced.parent
        parent_with_new_child.children[node_to_be_replaced.key] = replacement_node
        replacement_node.parent = parent_with_new_child

    # TODO Implement Mutation (We are not using it now so it's okay for now)
    def mutate(self):
        return self

    def get_output(self, resx, resy) -> np.array:

        xmax, ymax = self.get_pixel_range(resx, resy)
        xmin = -xmax
        ymin = -ymax
        zmin, zmax = 0, 1

        xspace = np.linspace(xmin, xmax, resx)
        yspace = np.linspace(ymin, ymax, resy)
        zspace = np.linspace(zmin, zmax, self.params['n_channels'])
        zz, xx, yy = np.meshgrid(zspace, xspace, yspace, indexing='ij')
        # outs = np.zeros((self.params['n_channels'],resx, resy))
        out = self.tree.evaluate(xx, yy, zz)
        out = out * np.ones_like(zz)  # Sometimes out is a single scalar value. This handles that
        for i in range(self.params['n_channels']):
            max_out, min_out = np.max(out[i]), np.min(out[i])
            out[i] = (out[i] - min_out) / ((max_out - min_out) + 0.00001)
        return out.swapaxes(0, 2)

    def evaluate(self):
        pass

    def tree_copy(self):
        return self.tree.copy(None)

    def get_statistics(self):
        return [self.tree.max_depth(), self.tree.number_of_nodes()]


class LSystem(Representation):
    def __init__(self, params):
        super().__init__(params)

    def crossover(self, other) -> List[Representation]:
        pass

    def mutate(self):
        pass

    def get_output(self, resx, resy) -> np.array:
        pass

    def evaluate(self):
        pass


class CellularAutomatron(Representation):
    animate = True  # does the representation move?
    go_back = True  # if the representation is animated, should the gif go back to the first frame on display?

    def __str__(self):
        return str(self.past_grid)

    def __init__(self, params):
        super().__init__(params)

        self.size = params['ca_params']['size']

        # Initialize grid
        row = np.array([0 if np.random.rand() < 0.15 else 1 for _ in range(self.size)])
        self.grid = np.array([np.random.permutation(row) for _ in range(self.size)])
        self.grid = np.stack(self.grid)
        self.grid = np.pad(self.grid, 1, 'constant')
        self.past_grid = self.grid.copy()

        # Create a dictionary of update rules
        all_combs = np.array([np.reshape(np.array(i), (3, 3)) for i in itertools.product([0, 1], repeat=3 * 3)])

        # Initialize with game of life rules
        # self.update_rules = {k.tobytes(): self.game_of_life(k) for k in all_combs}

        # Initialize with random rules
        keys = [k.tobytes() for k in all_combs]
        values = [1 if np.random.rand() <= 0.1 else 0 for _ in range(len(keys))]
        self.update_rules = dict(zip(keys, values))
        self.get_output(resx=64, resy=64)

    def crossover(self, other):
        cross_point = np.random.randint(low=0, high=len(self.update_rules))
        keys = list(self.update_rules)[cross_point:]
        self_values = list(self.update_rules.values())[cross_point:]
        other_values = list(other.update_rules.values())[cross_point:]
        self.update_rules.update(dict(zip(keys, other_values)))
        other.update_rules.update(dict(zip(keys, self_values)))
        return [self, other]

    def mutate(self):
        self.update_rules.update((x, not v) for x, v in self.update_rules.items() if np.random.rand() < 0.1)
        return self

    def get_output(self, resx=64, resy=64) -> np.array:
        self.update(1)
        avg = (self.grid + self.past_grid) / 2
        avg = avg[1:-1, 1:-1]
        output = np.dstack((avg, avg, avg))
        self.past_grid = self.grid.copy()
        return output

    def evaluate(self):
        return 1

    def game_of_life(self, k):
        cell = k[1, 1]
        alive_neigh = np.sum(k) - cell
        return 1 if cell and alive_neigh == 2 or alive_neigh == 3 else 0

    def update(self, n):
        for _ in range(n):
            for x in range(self.size):
                for y in range(self.size):
                    self.grid[x + 1, y + 1] = self.update_rules[self.grid[x:x + 3, y:y + 3].tobytes()]


class KernelCA(Representation):
    animate = True

    def __str__(self):
        return str(self.world)

    def __init__(self, params):
        super().__init__(params)
        self.soft_zero = 1e-10
        self.space_res = params['dl_params']['space_res']
        self.state_res = params['dl_params']['state_res']
        self.time_res = params['dl_params']['time_res']

        # Game of Life params
        # self.dx = 1
        # self.mu = 0.35
        # self.sigma = 0.07
        # self.dt = 1
        # self.beta = np.array([1])
        # self.radius = 2
        # self.kernel_core = lambda r, q=1/4: (r >= q) * (r <= 1 - q) + (r < q) * 0.5
        # self.growth_mapping = lambda pot: np.exp(- (pot - self.mu) ** 2 / (2 * self.sigma ** 2)) * 2 - 1

        # Other params
        self.dx = 1 / self.space_res
        self.dt = 1 / self.time_res
        self.dp = 1 / self.state_res

        self.mu = uniform(0.01, 0.75)
        self.sigma = uniform(0.1, 0.9 - self.mu)
        self.beta = np.clip(rand(randint(1, 8)), 0.1, 1)  # kernel peaks
        self.alpha = randint(2, 10)
        self.radius = randint(2, 60)

        # Texture of the kernel
        self.kernel_cores = {
            0: lambda r: (4 * r * (1 - r)) ** self.alpha,
            1: lambda r: np.exp(self.alpha - self.alpha / (4 * r * (1 - r) + self.soft_zero))
        }

        self.growth_mappings = {
            0: lambda pot: 2 * (pot >= self.mu - 3 * self.sigma) * (pot < self.mu + 3 * self.sigma) * (
                    1 - (pot - self.mu) ** 2 / (9 * self.sigma ** 2)) ** self.alpha - 1,
            1: lambda pot: 2 * np.exp(- (pot - self.mu) ** 2 / (2 * self.sigma ** 2)) - 1
        }

        self.kc_i, self.gm_i = randint(2), randint(2)
        self.kernel_core = self.kernel_cores[self.kc_i]
        self.growth_mapping = self.growth_mappings[self.gm_i]

        self.kernel, self.kernel_fft = self.pre_calc_kernel()
        self.world = self.get_world()

        self.run_automaton()

    def pre_calc_kernel(self):
        i = np.array([np.arange(self.space_res), ] * self.space_res)
        x = (i - int(self.space_res / 2)) / self.radius
        y = x.T
        dist = np.sqrt(x ** 2 + y ** 2)
        kernel = self.kernel_shell(dist)
        kernel_norm = kernel / np.sum(kernel)
        kernel_fft = fft2(kernel_norm)
        return kernel, kernel_fft

    def re_calc_kernel(self):
        self.kernel, self.kernel_fft = self.pre_calc_kernel()

    # Skeleton of the kernel
    def kernel_shell(self, r):
        b_size = len(self.beta)
        b_r = b_size * r
        b = self.beta[np.minimum(np.floor(b_r).astype(int), b_size - 1)]
        return (r < 1) * self.kernel_core(np.minimum(b_r % 1, 1)) * b

    def get_world(self):
        return randint(self.state_res + 1, size=(self.space_res, self.space_res))

    def run_automaton(self):
        world_fft = fft2(self.world)
        pot_fft = np.multiply(self.kernel_fft, world_fft)
        pot = fftshift(ifft2(pot_fft).real)
        growth = self.growth_mapping(pot)
        self.world = np.clip(self.world + self.dt * growth, 0, 1)
        return pot, growth

    def crossover(self, other):
        # crossover of betas
        cross_p = randint(1, min((self.beta.size, other.beta.size)) + 1)
        temp = self.beta[cross_p:]
        self.beta = np.append(self.beta[:cross_p], other.beta[cross_p:])
        other.beta = np.append(other.beta[:cross_p], temp)

        # crossover of radii
        temp_radius = self.radius
        self.radius = other.radius
        other.radius = temp_radius

        self.re_calc_kernel()
        other.re_calc_kernel()

        return [self, other]

    def mutate(self):
        self.mu = np.clip(self.mu + choice((-1, 1)) * normal(self.mu, 0.3), 0.01, 0.75)
        self.sigma = uniform(0.1, 0.9 - self.mu)
        self.beta[randint(self.beta.size)] = np.clip(rand(), 0.1, 1)
        self.radius = randint(2, 60)
        self.re_calc_kernel()
        return self

    def get_output(self, resx=256, resy=256) -> np.array:
        pot, growth = self.run_automaton()
        return np.dstack((self.world, pot, growth))

    def evaluate(self):
        return np.var(self.world)


class MantasMakelisRepresentation(Representation):
    def __init__(self, params):
        super().__init__(params)
        self.image = Image.open('data/mantas2.jpg')
        self.image = np.array(self.image)
        # self.image = self.image + random.normal(0, 1, self.image.shape) * 0.01

    def crossover(self, other):
        return [MantasMakelisRepresentation(self.params), MantasMakelisRepresentation(self.params)]

    def mutate(self):
        return self

    def get_output(self, resx, resy) -> np.array:
        return self.image

    def evaluate(self):
        return 1
