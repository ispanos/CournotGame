# Cournot market competition simulations

This python script can calculate the amount of production of N companies, in a Cournot game market.

At the time of writing this description, the following assumptions must be made:

- The demand is linear and the "demand" parameter given in the script is the inverse demand.
- The companies that compete have a quadratic cost curves. The parameter given to the script is the
marginal cost of the company.

In mergers:

- In edge cases where one company's marginal cost is linear, but the other one's are constant, the script exits. Such case would increase the complexity of the script. If I get a single reason I should at this, I will.

Please see the [github page for this repository](https://ispanos.github.io/CournotGame/) for more information.

Breaking changes might occur once I start working on Isoelastic Demand Curves.
Feature updates will be prioritized before documentation updates.
Please download a release of this script for the proper documentation.
