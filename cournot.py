# Released under MIT License
# Copyright (c) 2021 Spanos Ioannis, github.com/ispanos

# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
# USE OR OTHER DEALINGS IN THE SOFTWARE.

# NOTE: Most times Demand is mentioned, it's the inverse demand

import numpy as np
import pandas as pd
from typing import List, Tuple  # , Annotated
import copy
import statsmodels.api as sm


# import matplotlib.pyplot as plt


def infinite_sequence():
    num = 1
    while True:
        yield str(num)
        num += 1


name_sequence = infinite_sequence()


def next_name():
    return next(name_sequence)


def reset_names():
    global name_sequence
    name_sequence = infinite_sequence()


class Company:

    def __init__(self, i: float, s: float, name: str = None):
        self._i = float(i)
        self._s = float(s)
        self._name = name
        self._production = None
        self._regression_data = None

        if not name:
            self._name = next_name()

    @property
    def i(self):
        """
        The intercept of the estimated marginal cost curve
        Type: float
        """
        return self._i

    @property
    def s(self):
        """
        The slope of the estimated marginal cost curve
        Type: float
        """
        return self._s

    @property
    def equation(self) -> str:
        """
        The slope of the estimated marginal cost curve
        Type: float
        """
        return f"Mc = {str(round(self._i, 2))} + {str(round(self._s, 2))} * q"

    @property
    def full_equation(self) -> str:
        """
        The slope of the estimated marginal cost curve
        Type: float
        """
        return f"Mc = {str(self._i)} + {str(self._s)} * q"

    @property
    def name(self):
        """
        A name to identify the company
        Type: string
        """
        return self._name

    @property
    def production(self):
        """
        A name to identify the company
        Type: float
        """
        return self._production

    @property
    def regression_data(self):
        """
        Regression data from OLS regression.
        """
        return self._regression_data

    def set_prod(self, production: float):
        """
        Set the production of the company in the current market
        """
        self._production = production
        return self

    def set_name(self, name: str):
        """
        Set a name to identify the company by
        """
        self._name = name
        return self

    def profits(self, p: float):
        """
        The profits of the company
        Type: float
        """
        return (p * self._production) - (self._i +
                                         self._s * self._production
                                         ) * self._production

    def set_regression_data(self, stuff):
        self._regression_data = stuff
        return self


Demand = Tuple[float, float]
CompanyList = List[Company]


def calculate_price(total_q: float, demand: Demand) -> float:
    """
    Calculates the equilibrium price, given a linear Demand curve and
    the total units produced by the companies

    Args:
      total_q: The total units produced
      demand: The parameters of a linear demand curve

    Returns:
      The equilibrium price

    """
    return demand[0] - demand[1] * total_q


def set_cournot_production(demand: Demand,
                           companies: CompanyList) -> CompanyList:
    """
    Return a list with the addition of the production units in
    every company in the given list, when all companies are in a
    Cournot competition.
    Args:
        demand: Market's demand
        companies: A list of companies

    Returns: Company_List with updated production values

    """
    # Create an array of length N -> (M_i + 2 * B)
    diagonal: List[float] = [x.s + 2 * demand[1] for x in companies]

    dimension = len(companies)

    # Create a matrix of N x N dimension filled with B
    x = np.full((dimension, dimension), demand[1], dtype=float)

    # Replace the diagonal of the matrix with (M_i + 2 * B)
    # This creates matrix named H in the documentation above
    # noinspection PyTypeChecker
    np.fill_diagonal(x, diagonal)

    # Create a matrix N x 1 with ( A - K ) -- Named U in the documentation above
    constants = [demand[0] - comp.i for comp in companies]

    # Our solution is an array of quantities, length N.
    productions = np.linalg.solve(x, constants).flatten()

    for i, c in enumerate(companies):
        c.set_prod(productions[i])

    return companies


def merge_companies(comp_i: Company, comp_j: Company) -> Company:
    """
    Merges two companies by horizontally adding their production output curves
    in relation with their marginal costs.

    Args:
        comp_i: Company that will merge with another one
        comp_j: Company that will merge with another one
    Returns:
        Company post merge
    """
    if (comp_i.s + comp_j.s) == 0:
        new_comp = Company(min(comp_i.i, comp_j.i), 0)
    elif comp_i.s == 0 or comp_j.s == 0:
        print("Edge case is ot accounted for.")
        exit()
    else:
        new_comp = Company((comp_j.s * comp_i.i + comp_i.s * comp_j.i) /
                           (comp_i.s + comp_j.s),
                           comp_i.s * comp_j.s / (comp_i.s + comp_j.s),
                           comp_i.name + '&' + comp_j.name)
    return new_comp


def merge_two(demand: Demand, companies: CompanyList, to_merge: Tuple[int, int]):
    """
    Replace the two companies that merge, in the given list, with the newly formed one.
    Args:
        demand: Market's demand
        companies: An ordered list of the companies that are competing
        to_merge: A tuple composed of the two indexes of the two companies that
                will merge.
    Returns:
        Company_List after given merger
    """
    companies_post_merge = copy.copy(companies)

    comp_i = companies_post_merge[to_merge[0]]
    comp_j = companies_post_merge[to_merge[1]]
    new_company = merge_companies(comp_i, comp_j)

    companies_post_merge.remove(comp_i)
    companies_post_merge.remove(comp_j)
    companies_post_merge.insert(0, new_company)
    return set_cournot_production(demand, companies_post_merge)


def market_stats_dump(companies: CompanyList, q: float, p: float):
    """
    Print data for the market.
    """
    for comp in companies:
        print(f"Company {comp.name} with {comp.equation}\n",
              f"\tProduces {round(comp.production, 2)}",
              f" with €{round(comp.profits(p), 2)} profit.\n")

    print(f"Total production is {round(q, 2)} units @ €{round(p, 2)}.")


def hhi(c: CompanyList) -> int:
    """
    Herfindahl-Hirschman Index
    Args:
        c: List of companies

    Returns: Herfindahl-Hirschman Index

    """
    q_tot = sum([x.production for x in c])
    return int(round(sum([(100 * x.production / q_tot) ** 2 for x in c])))


def regress_info(array_x, array_y):
    """
    Runs OLS for x, y arrays.

    Args:
        array_x: Independent variable
        array_y: Dependent variable

    Returns: sm.OLS.fit

    """
    array_x = sm.add_constant(array_x)
    model = sm.OLS(array_y, array_x)
    return model.fit()


def create_est_company(model: sm.OLS.fit) -> Company:
    """
    Given the OLS.fit date, create a new Company

    Args:
        model: OLS regression data of Q and MC arrays

    Returns: Company

    """
    new_company = Company(model.params[0], model.params[1])
    new_company.set_regression_data(model)
    return new_company


def estimate_comp_productions(
        demand: Demand,
        marginal_costs: Tuple[pd.Series, pd.Series, pd.Series]
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    WORK IN PROGRESS -- NOT OPTIMIZED
    Returns the production for each company, calculated by their
    marginal costs and demand curve. Only works for 3 companies
    """
    c1, c2, c3 = marginal_costs
    q1, q2, q3 = [], [], []
    for i in range(len(c1)):
        x = np.full((3, 3), .5, dtype=float)

        # Replace the diagonal of the matrix with (M_i + 2 * B)
        # This creates matrix named H in the documentation above
        # noinspection PyTypeChecker
        np.fill_diagonal(x, np.ones((3,), dtype=float))

        # Create a matrix MC_i / 2*B
        constants = [(demand[0] - c1[i]) / demand[1],
                     (demand[0] - c2[i]) / demand[1],
                     (demand[0] - c3[i]) / demand[1]]

        # Our solution is an array of quantities, length N.
        productions = np.linalg.solve(x, constants).tolist()
        q1.append(productions[0])
        q2.append(productions[1])
        q3.append(productions[2])

    pd_q1 = pd.Series(q1, name='q1')
    pd_q2 = pd.Series(q2, name='q2')
    pd_q3 = pd.Series(q3, name='q3')

    return pd_q1, pd_q2, pd_q3


def estimate_curves_alt(file):
    """
    WORK IN PROGRESS -- NOT OPTIMIZED
    Given a file with data in a specific format - required by my project,
    returns a Demand and Company_List.

    Args:
        file : Requires a file with columns 'P', 'Q', 'C1',
        'C2' ,'C3'. The combinations of 'P' and 'Q' are market
        equilibrium points. 'C1','C2','C3' are the total costs of
        the companies at 'Q' level of production.

    Returns:
        Demand: The parameters of the inverse demand in the current market
        Company_List: The parameters of the marginal cost for the given Companies.
    """

    companies: CompanyList = []

    # noinspection PyBroadException
    try:
        df = pd.read_csv(file)
        total_prod: pd.Series = df['Q']
        prices: pd.Series = df['P']
        total_costs = (df['C1'], df['C2'], df['C3'])

        ols_demand = regress_info(prices, total_prod)
        a, b = ols_demand.params[0], ols_demand.params[1]
        demand_estimate: Demand = (abs(a / b), 1 / abs(b))

        for total_cost in total_costs:
            marginal_costs = total_cost.diff() / total_prod.diff()
            marginal_costs.name = f"M{total_cost.name}"
            df[f"M{total_cost.name}"] = marginal_costs
            # all_marginal_costs.append(marginal_costs)

        df['q1'], df['q2'], df['q3'] = estimate_comp_productions(
            demand_estimate, (df['MC1'], df['MC2'], df['MC3']))

        for company_prod, marginal_costs in ((df['q1'], df['MC1']),
                                             (df['q2'], df['MC2']),
                                             (df['q3'], df['MC3'])):
            ols_mc = regress_info(company_prod[1:], marginal_costs[1:])
            companies.append(create_est_company(ols_mc))

        return ols_demand, demand_estimate, companies

    except Exception:
        print("Failed while importing data")
        exit()


def estimate_curves(file):
    """
    Given a file with data in a specific format - required by my project,
    returns a Demand and Company_List.

    Args:
        file : Requires a file with columns 'P', 'Q', 'C1',
        'C2' ,'C3'. The combinations of 'P' and 'Q' are market
        equilibrium points. 'C1','C2','C3' are the total costs of
        the companies at 'Q' level of production. We assume those
        companies have no fixed costs when calculating the marginal
        costs.

    Returns:
        Demand: The parameters of the inverse demand in the current market
        Company_List: The parameters of the marginal cost for the given Companies.
    """

    companies: CompanyList = []

    # noinspection PyBroadException
    try:
        df = pd.read_csv(file)
        total_prod: pd.Series = df['Q']
        prices: pd.Series = df['P']
        total_costs = (df['C1'], df['C2'], df['C3'])

        for total_cost in total_costs:
            marginal_costs = total_cost.diff() / total_prod.diff()
            marginal_costs.name = f"M{total_cost.name}"

            ols_mc = regress_info(total_prod[1:], marginal_costs[1:])
            companies.append(create_est_company(ols_mc))

        ols_demand = regress_info(prices, total_prod)
        a, b = ols_demand.params[0], ols_demand.params[1]
        demand_estimate: Demand = (abs(a / b), 1 / abs(b))
        return ols_demand, demand_estimate, companies
    except Exception:
        print("Failed while importing data")
        exit(1)


def consecutive_merger(old_price, companies, combination, demand):
    post_merge = merge_two(demand, companies, combination)
    new_quantity = sum([comp.production for comp in post_merge])
    new_price = calculate_price(new_quantity, demand)

    i, j = combination[0], combination[1]
    old_profits = sum([companies[i].profits(old_price),
                       companies[j].profits(old_price)])

    print(f"The sum of the profits, of companies {companies[i].name}",
          f"and {companies[j].name}\n \tbefore the merger, were:",
          f"€{round(old_profits, 2)}\n")
    market_stats_dump(post_merge, new_quantity, new_price)
    print(f"HHI:{hhi(post_merge)}")
    print(f"The new price is {round(((new_price - old_price) * 100) / old_price)}% higher.")
    print(("\n" + "*" * 60 + "\n"))

    return new_price, post_merge


def main(file, q_is_sum=False):
    if q_is_sum and isinstance(file, str):
        ols_demand, estimated_demand, companies = estimate_curves_alt(file)
    elif isinstance(file, str):
        ols_demand, estimated_demand, companies = estimate_curves(file)
    else:
        estimated_demand, companies = file

    a, b = estimated_demand[0], estimated_demand[1]
    companies_b4merge = set_cournot_production(estimated_demand, companies)

    quantity = sum([comp.production for comp in companies_b4merge])
    price = calculate_price(quantity, estimated_demand)

    if ols_demand := None:
        print(ols_demand.summary())

    # print(f"The demand curve is: Q = {round(a,2)} - {abs(round(b,2))} * P")
    print(
        f"The demand curve is: Q = {round(abs(a / b), 2)} - {round(1 / abs(b), 2)} * P",
        f"< = > P = {round(a, 2)} - {round(b, 2)} * Q")

    for c in companies:
        if c.regression_data:
            print(c.regression_data.summary(), "\n" * 5)

    print(("*" * 60 + "\n"))
    print(("*" * 60 + "\n"))
    market_stats_dump(companies_b4merge, quantity, price)
    print(f"HHI:{hhi(companies_b4merge)}")
    print(("\n" + "*" * 60 + "\n"))

    for combination in [(0, 1), (0, 2), (1, 2)]:
        print(("*" * 60 + "\n"))
        post_merge = merge_two(estimated_demand, companies_b4merge, combination)

        quantity = sum([comp.production for comp in post_merge])
        new_price = calculate_price(quantity, estimated_demand)

        market_stats_dump(post_merge, quantity, new_price)
        print(f"HHI:{hhi(post_merge)}")
        print(("\n" + "*" * 60 + "\n"))
    print(("*" * 60 + "\n"))

    # for company in companies:
    #     print(f"Company {company.name} - {company.full_equation}")


if __name__ == "__main__":
    # Using Q as the total Q, as well as the Q for each company
    # main('./data.csv', False)

    # Using Q as the total Q and calculating q_i by the MC_i and Q
    # main('./out.csv', True)

    # No regression
    D = (2221.08, 15.81)
    C: CompanyList = [Company(2.71, 5.34),
                      Company(6.13, 1.11),
                      Company(4.75, 1.53)]
    data = (D, C)
    main(data)
