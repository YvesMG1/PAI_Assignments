"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel, Sum, Product, DotProduct
# import additional ...


# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    # TODO: implement a self-contained solution in the BO_algo class.

    def __init__(
            # surrogate model to approximate f which we will optimize
            # surrogate to the SA (synthetic availability)
            # beta tradeoff parameter for ucb acquisition function
            # lambda for the Lagrangian penalty # 0.5 * Matern(nu=2.5, length_scale=0.5) + ConstantKernel(
        # 4.0) + DotProduct(sigma_0=0) + Matern(nu=2.5)
            self, surrogate_model=GaussianProcessRegressor(
                kernel=0.5 * RBF(length_scale=1.0), n_restarts_optimizer=10, alpha=0.15**2),
            surrogate_SA=GaussianProcessRegressor(kernel=ConstantKernel(
                4.0) + DotProduct(sigma_0=0) + Matern(nu=2.5), n_restarts_optimizer=10, alpha=0.0001**2),
            beta=2,  # beta 2 best score with no decay so far
            lamb=1  # lamb 1 best score with increment 0.2 so far
    ):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.

        self.lamb = lamb
        self.beta = beta
        self.surrogate_model = surrogate_model
        self.surrogate_SA = surrogate_SA
        self.lamb_increment = 0.2  # 0.2

        self.data_points = []  # list to store sampled data points

        pass

    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.

        recommendation = self.optimize_acquisition_function()
        recommendation = np.array(recommendation).reshape(1, 1)

        self.lamb += self.lamb_increment

        return recommendation

    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # TODO: Implement the acquisition function you want to optimize.

        # going to try with UCB acquisition function
        mean, std = self.surrogate_model.predict(x, return_std=True)

        ucb = mean + self.beta * std  # Calculate UCB

        sa_mean, sa_std = self.surrogate_SA.predict(x, return_std=True)

        # Calculate probability of surrogate being feasible
        # This is the probability of the constraint being satisfied
        # constrain is satisfied if SA < 4
        pf_x = norm.cdf(4, loc=sa_mean, scale=sa_std)

        # punish on constraint more over time
        # punish low pf_x more over time and high pf_x less over time
        pf_x = pf_x ** self.lamb

        if np.isnan(pf_x):
            print("pf_x is nan")
            return ucb
        else:
            return ucb * pf_x

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.

        # add data point to the list of data points
        if isinstance(x, np.ndarray):
            x = x.item()

        self.data_points.append((x, f, v))

        X, Y, C = zip(*self.data_points)

        X = np.array(X).reshape(-1, 1)
        Y = np.array(Y).reshape(-1, 1)
        C = np.array(C).reshape(-1, 1)

        # Fit the Gaussian Process model to the data
        self.surrogate_model.fit(X, Y)
        self.surrogate_SA.fit(X, C)

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.

        # testing function at 10,000 points within the domain
        candidate_points = np.linspace(0, 10, num=10000)

        # get predictions on the candidate points
        mean_predictions = self.surrogate_model.predict(
            candidate_points.reshape(-1, 1), return_std=False)

        # penalise predictions where SA is estimated to be > 4
        SA_predictions = self.surrogate_SA.predict(
            candidate_points.reshape(-1, 1), return_std=False)
        lagrangian_penalties = self.lamb * np.maximum(SA_predictions - 4, 0)
        penalised_predictions = mean_predictions - lagrangian_penalties

        # get input that had the highest prediction
        best_candidate = candidate_points[np.argmax(penalised_predictions)]

        # ensure best candidate is indeed between zero and ten, in case of numerical discrepancies
        solution = max(min(best_candidate, 10), 0)

        return solution

      #  raise NotImplementedError

    def plot(self, plot_recommendation: bool = True):
        """Plot objective and constraint posterior for debugging (OPTIONAL).

        Parameters
        ----------
        plot_recommendation: bool
            Plots the recommended point if True.
        """
        pass

# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---


def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    return x_init


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.randn()
        cost_val = v(x) + np.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')


if __name__ == "__main__":
    main()
