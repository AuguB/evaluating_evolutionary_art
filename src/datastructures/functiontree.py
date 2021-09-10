from numpy import random
import numpy as np


class FunctionTree:
    def __str__(self):
        pass

    def __init__(self, params, func_idx, parent=None, key=None):
        self.key = key
        self.params = params
        self.func_idx = func_idx
        self.depth = 0 if not parent else parent.depth + 1
        self.parent: FunctionTree = parent
        self.children = {}
        self.make_children()

    def make_children(self):
        pass

    def sample_child(self):
        if self.params['method'] == 'full':
            if self.depth < self.params['d_max']:
                idx = random.randint(low=0, high=len(self.params['branch_samples']))
                if idx < self.params['n_un']:
                    return UnaryBranch, idx
                else:
                    return BinaryBranch, idx
            else:
                return Leaf, random.randint(low=0, high=len(self.params['leaf_samples']))
        elif self.params['method'] == 'grow':
            if self.depth < self.params['d_max']:
                idx = random.randint(low=0, high=len(self.params['branch_samples']) + self.params['n_t'])
                if idx < self.params['n_un']:
                    return UnaryBranch, idx
                elif idx < self.params['n_un'] + self.params['n_bin']:
                    return BinaryBranch, idx
                else:
                    return Leaf, idx - len(self.params['branch_samples'])
            else:
                return Leaf, random.randint(low=0, high=len(self.params['leaf_samples']))

    def evaluate(self, x, y, z=None):
        pass

    def copy(self, new_parent):
        pass

    def max_depth(self):
        pass

    def number_of_nodes(self):
        return 1 + sum([i.number_of_nodes() for i in self.children.values()])

    def get_nth_node(self, n):
        pass


class BinaryBranch(FunctionTree):

    def __str__(self):
        return f"({str(self.children['left'])}{self.funcname}{str(self.children['right'])})"

    def __init__(self, params, func_idx, parent=None, key=None):
        super().__init__(params, func_idx, parent, key)
        self.func = self.params['branch_samples'][self.func_idx][0]
        self.funcname = self.params['branch_samples'][self.func_idx][1]

    def make_children(self):
        branch_l, idx_l = self.sample_child()
        self.children['left'] = branch_l(self.params, idx_l, self, 'left')
        branch_r, idx_r = self.sample_child()
        self.children['right'] = branch_r(self.params, idx_r, self, 'right')

    def evaluate(self, x, y, z=None):
        return self.func(self.children['left'].evaluate(x, y, z), self.children['right'].evaluate(x, y, z))

    def copy(self, new_parent):
        copy = BinaryBranch(self.params, self.func_idx, new_parent, key=self.key)
        copy.children = {}
        for c in self.children.keys():
            copy.children[c] = self.children[c].copy(copy)
        return copy

    def max_depth(self):
        return max(self.children['left'].max_depth(), self.children['right'].max_depth())

    def get_nth_node(self, n):
        if n == 0:
            return self
        elif n - 1 < self.children['left'].number_of_nodes():
            return self.children['left'].get_nth_node(n - 1)
        else:
            return self.children['right'].get_nth_node(n - 1 - self.children['left'].number_of_nodes())


class UnaryBranch(FunctionTree):
    def __str__(self):
        funcname = self.params['branch_samples'][self.func_idx][1]
        return f"{funcname}({str(self.children['middle'])})"

    def __init__(self, params, func_idx, parent=None, key=None):
        super().__init__(params, func_idx, parent, key)
        self.func = self.params['branch_samples'][self.func_idx][0]
        self.funcname = self.params['branch_samples'][self.func_idx][1]

    def make_children(self):
        branch_m, idx_m = self.sample_child()
        self.children['middle'] = branch_m(self.params, idx_m, self, 'middle')

    def evaluate(self, x, y, z=None):
        return self.func(self.children['middle'].evaluate(x, y, z))

    def copy(self, new_parent):
        copy = UnaryBranch(self.params, self.func_idx, new_parent, key=self.key)
        copy.children = {}
        copy.children['middle'] = self.children['middle'].copy(copy)
        return copy

    def max_depth(self):
        return self.children['middle'].max_depth()

    def get_nth_node(self, n):
        if n == 0:
            return self
        else:
            return self.children['middle'].get_nth_node(n - 1)


class Leaf(FunctionTree):
    def __str__(self):
        if not self.func in ['x', 'y', 'z']:
            return str(np.round(self.func, 2))
        return self.func

    def __init__(self, params, func_idx, parent=None, key=None):
        super().__init__(params, func_idx, parent, key)
        self.func = self.params['leaf_samples'][self.func_idx]

    def make_children(self):
        pass

    def evaluate(self, x, y, z=None):
        if self.func == 'x':
            return x
        elif self.func == 'y':
            return y
        elif self.func == 'z':
            return z
        else:
            return self.func

    def copy(self, new_parent):
        copy = Leaf(self.params, self.func_idx, new_parent, key=self.key)
        copy.children = {}
        return copy

    def max_depth(self):
        return self.depth

    def get_nth_node(self, n):
        return self
