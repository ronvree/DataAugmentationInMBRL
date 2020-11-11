

if __name__ == '__main__':
    import scipy
    from scipy import stats
    import pandas as pd

    path = './pendulum_values.csv'

    num_samples = 9

    df = pd.read_csv(path, index_col=0)

    xs_0 = df.loc['no aug'][:num_samples].values
    xs_1 = df.loc['translate'][:num_samples].values
    xs_2 = df.loc['mirror'][:num_samples].values

    print(scipy.stats.ttest_ind(xs_0, xs_1, equal_var=False))


