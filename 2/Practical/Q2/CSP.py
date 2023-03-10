import numpy as np
import numpy.typing as npt

class CSP:
    def __init__(self, n: int):
        """
        Here we initialize all the required variables for the CSP computation,
        according to the n.
        """
        # Your code here
        self.n = n
        self.grid = np.array(np.zeros(shape=(n, n), dtype=np.int8))
        self.domain = np.array(np.ones(shape=(n, n), dtype=np.int8))
        self.number_of_iteration = 0

    def check_constraints(self) -> bool:
        """
        Here we check the grid horizontally, vertically and diagonally
        """
        # At first row and columns
        if np.max(np.sum(self.grid, axis=0)) > 1:
            return False
        if np.max(np.sum(self.grid, axis=1)) > 1:
            return False
        # Now check diags
        queens: list[tuple[int, int]] = [] # List of queens in screen
        for a in range(self.n):
            for b in range(self.n):
                if self.grid[a, b] == 1:
                    queens.append((a, b))
        for queen1 in queens:
            for queen2 in queens:
                if queen1 == queen2:
                    continue
                if queen1[0] + queen1[1] == queen2[0] + queen2[1] or queen1[0] - queen1[1] == queen2[0] - queen2[1]:
                    return False
        return True

    @staticmethod
    def fill_cross(a: npt.NDArray[np.int8], row: int, col: int, newval: int):
        # From https://stackoverflow.com/a/56488480/4213397
        # I have no idea what the fuck this is.
        # All i know is that this function will diagonally fill the domains
        n = len(a)
        if row + col >= n:
            anti_diag_start = (row+col-n+1,n-1)
        else:
            anti_diag_start = (0,row+col)

        if row > col:
            diag_start = (row-col,0)
        else:
            diag_start = (0,col-row)

        r, c = [np.ravel_multi_index(i,a.shape) for i in [diag_start, anti_diag_start]]
        a.ravel()[r:r+(n-diag_start[0]-diag_start[1])*(n+1):n+1] = newval
        a.ravel()[c:c*(n+1):n-1] = newval
        return a

    def forward_check(self, i: int, j: int, set: bool):
        """
        After assigning a queen to ith column, we can make a forward check
        to boost up the computing speed and prune our search tree.
        """
        to_fill_with_value = (1 if set else 0)
        to_fill_with = np.full(shape=self.n, fill_value=to_fill_with_value , dtype=np.int8)
        self.domain[i,:] = to_fill_with
        self.domain[:,j] = to_fill_with
        self.domain = CSP.fill_cross(self.domain, i, j, to_fill_with_value)
        pass
        
    def _solve_problem_with_backtrack(self, i: int) -> bool:
        """
         In this function we should set the ith queen in ith column and call itself recursively to solve the problem.
        """
        next_tile = i+1
        for candidate in range(self.n):
            self.grid[i, candidate] = 1
            self.number_of_iteration += 1
            if self.check_constraints():
                if next_tile == self.n: # we have checked everything
                    return True
                if self._solve_problem_with_backtrack(i+1): # Move to next queen
                    return True
            self.grid[i, candidate] = 0
        return False
        

    def solve_problem_with_backtrack(self):
        """
         In this function we should set the ith queen in ith column and call itself recursively to solve the problem
         and return solution's grid
        """
        self._solve_problem_with_backtrack(0)
        return self.grid

    def _solve_problem_with_forward_check(self, i: int):
        """
         In this function we should set the ith queen in ith column and call itself recursively to solve the problem.
        """
        next_tile = i+1
        for candidate in range(self.n):
            if self.domain[i, candidate] == 0: # not in domain
                continue
            self.forward_check(i, candidate, False)
            self.grid[i, candidate] = 1
            self.number_of_iteration += 1
            if next_tile == self.n: # we have checked everything
                return True
            if self._solve_problem_with_backtrack(i+1): # Move to next queen
                return True
            self.grid[i, candidate] = 0
            self.forward_check(i, candidate, True) # Restore domains

        return False

    def solve_problem_with_forward_check(self):
        """
         In this function we should set the ith queen in ith column and call itself recursively to solve the problem
         and return solution's grid
        """
        self._solve_problem_with_forward_check(0)
        return self.grid
