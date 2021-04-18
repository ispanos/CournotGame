import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import cournot


def generate_data(c1, c2, c3, d):
    sample_size = 33
    mu, sigma = 100, 3

    random_array = np.random.normal(mu, sigma, sample_size)

    a , b = d[0],d[1]

    time = pd.Series(list(range(1,sample_size+1)), name = 'T', dtype=int)


    production_sample = pd.Series(
                    [round(q,3) for q in random_array],
                     name = "Q")

    price_sample = pd.Series(
                [round((a - b * q),2) for q in production_sample],
                 name = 'P')

    df = pd.concat([time,price_sample,production_sample],axis=1)
    tot_q = sum([c1[2],c2[2],c3[2]])

    for i, c in enumerate((c1,c2,c3)):
        cost = []
        rand = np.random.normal(0,3)
        weight = c[2]/tot_q
        for prod in production_sample:
            q = weight*prod
            cost.append(((c[0] + c[1]*q) * q) + rand)
        
        df[f"C{i+1}"] = pd.Series(cost)

    return df


if __name__ == '__main__':
    #MC =  k   +  m  *  q
    c1 = [2.71, 5.34, 29.0]
    c2 = [6.13,  1.11, 36.36]
    c3 = [4.75,  1.53, 35.56]
    
    # P = 2221.08 - 15.81 * Q
    d = (2221.08, 15.81)

    df = generate_data(c1, c2, c3, d)
    # print(df)
    df.to_csv('out2.csv',index=False)


    cournot.main('out2.csv', True)

    # Expected values

    # The demand curve is: Q = 140.49 - 0.06 * P < = >
    # P = 2221.08 - 15.81 * Q
    # Company 1 -- Mc = 2.71 + 5.34 * q
    # Company 2 -- Mc = 6.13 + 1.11 * q
    # Company 3 -- Mc = 4.75 + 1.53 * q

    # Company 1 with Mc = 2.71 + 5.34 * q
    #         Produces 29.25  with €13529.37 profit.

    # Company 2 with Mc = 6.13 + 1.11 * q
    #         Produces 36.36  with €20906.58 profit.

    # Company 3 with Mc = 4.75 + 1.53 * q
    #         Produces 35.56  with €19995.47 profit.

    # Total production is 101.18 units @ €621.41.
