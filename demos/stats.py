

if __name__ == '__main__':
    import math
    import scipy
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    import pandas as pd
    from mlxtend.evaluate import permutation_test

    path = './pendulum_values.csv'

    num_samples = 20

    df = pd.read_csv(path, index_col=0)

    print(df.index)

    xs_0 = df.loc['no aug'][:num_samples].values
    xs_1 = df.loc['translate'][:num_samples].values
    xs_2 = df.loc['mirror'][:num_samples].values
    xs_3 = df.loc['mirror2'][:num_samples].values

    # xs_0[14] = -1000

    # print(xs_0[14])

    samples_1 = xs_0
    samples_2 = xs_2

    mean_1 = np.mean(samples_1)
    std_1 = np.std(samples_1)
    print('\nSeries 1')
    print(f'Mean: {mean_1}\n Std: {std_1}')

    mean_2 = np.mean(samples_2)
    std_2 = np.std(samples_2)
    print('\nSeries 2')
    print(f'Mean: {mean_2}\n Std: {std_2}')

    print()

    std_pool = math.sqrt((std_1 ** 2 + std_2 ** 2)/2)
    max_val = max(np.concatenate((samples_1, samples_2), axis=0))
    min_val = min(np.concatenate((samples_1, samples_2), axis=0))

    print(f'Effect size: {abs(max_val - min_val)/std_pool}')

    print('\nWelch')
    print(scipy.stats.ttest_ind(samples_1, samples_2, equal_var=False))
    print('\nPermutation')
    print(permutation_test(samples_1, samples_2, method='approximate', num_rounds=100000))
    print('\nLevene')
    print(scipy.stats.levene(samples_1, samples_2))

    # plt.hist(xs_1, bins=14)
    # plt.show()
