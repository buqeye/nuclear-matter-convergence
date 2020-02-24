import numpy as np
import pandas as pd


class InputData:

    def __init__(self, filename, Lambda, high_density=False):

        fits = {450: [1, 7], 500: [4, 10]}

        df = pd.read_csv(filename)
        if not high_density:
            # Convert differences to total prediction at each MBPT order
            mbpt_orders = ['Kin', 'MBPT_HF', 'MBPT_2', 'MBPT_3', 'MBPT_4']
            df[mbpt_orders] = df[mbpt_orders].apply(np.cumsum, axis=1)
            # 'total' is now unnecessary. Remove it.
            df.pop('total')

        self.df = df
        self.ref_2bf = 16
        self.Lambdas = df['Lambda'].unique()
        self.body2 = body2 = 'NN-only'
        self.body23 = body23 = 'NN+3N'

        if Lambda not in self.Lambdas:
            raise ValueError(f'Lambda must be in {self.Lambdas}')

        self.fit_n2lo, self.fit_n3lo = fits[Lambda]

        # Mask the DataFrame based on Lambda, n-body, and nucleon fraction values
        mask_fit = np.isin(df['fit'], fits[Lambda]) | np.isnan(df['fit'])

        mask_2bf = \
            (df['Body'] == body2) & \
            mask_fit & \
            (df['Lambda'] == Lambda)

        mask_23bf = \
            (df['Body'] == body23) & \
            mask_fit & \
            (df['Lambda'] == Lambda)

        df_n_2bf = df[mask_2bf & (df['x'] == 0)]
        df_s_2bf = df[mask_2bf & (df['x'] == 0.5)]
        df_n_2_plus_3bf = df[mask_23bf & (df['x'] == 0)]
        df_s_2_plus_3bf = df[mask_23bf & (df['x'] == 0.5)]

        # Setup kinematics
        self.kf_n = kf_n = df_n_2_plus_3bf[df_n_2_plus_3bf['OrderEFT'] == 'LO']['kf'].values
        self.kf_s = kf_s = df_s_2_plus_3bf[df_s_2_plus_3bf['OrderEFT'] == 'LO']['kf'].values
        self.density = df_s_2_plus_3bf[df_s_2_plus_3bf['OrderEFT'] == 'LO']['n'].values

        self.Kf_n = kf_n[:, None]
        self.Kf_s = kf_s[:, None]

        self.ref_n_3bf_vals = 16 * kf_n ** 3
        self.ref_s_3bf_vals = 16 * kf_s ** 3

        self.kf_s_dense = np.linspace(kf_s.min(), kf_s.max(), 100)
        self.Kf_s_dense = self.kf_s_dense[:, None]

        if not high_density:
            self.kf_avg = kf_avg = (kf_n + kf_s) / 2.
            self.Kf_avg = kf_avg[:, None]

            self.kf_n_dense = np.linspace(kf_n.min(), kf_n.max(), 100)
            self.Kf_n_dense = self.kf_n_dense[:, None]

            self.kf_avg_dense = np.linspace(kf_avg.min(), kf_avg.max(), 100)
            self.Kf_avg_dense = self.kf_avg_dense[:, None]

            self.ref_avg_3bf = 16 * kf_avg ** 3

        # ref_n_3bf = 8 * kf_n**6
        # ref_s_3bf = 8 * kf_s**6
        # ref_d_3bf = 8 * kf_d**6

        # Extract each type of observable
        self.y_n_2bf = y_n_2bf = np.array([
            df_n_2bf[df_n_2bf['OrderEFT'] == order]['MBPT_4'].values
            for order in df_n_2bf['OrderEFT'].unique()
        ]).T
        self.y_s_2bf = y_s_2bf = np.array([
            df_s_2bf[df_s_2bf['OrderEFT'] == order]['MBPT_4'].values
            for order in df_s_2bf['OrderEFT'].unique()
        ]).T
        if not high_density:
            self.y_d_2bf = y_n_2bf - y_s_2bf

        self.y_n_2_plus_3bf = y_n_2_plus_3bf = np.array([
            df_n_2_plus_3bf[df_n_2_plus_3bf['OrderEFT'] == order]['MBPT_4'].values
            for order in df_n_2_plus_3bf['OrderEFT'].unique()
        ]).T
        self.y_s_2_plus_3bf = y_s_2_plus_3bf = np.array([
            df_s_2_plus_3bf[df_s_2_plus_3bf['OrderEFT'] == order]['MBPT_4'].values
            for order in df_s_2_plus_3bf['OrderEFT'].unique()
        ]).T
        if not high_density:
            self.y_d_2_plus_3bf = y_n_2_plus_3bf - y_s_2_plus_3bf

            self.y_n_3bf = y_n_2_plus_3bf - y_n_2bf
            self.y_s_3bf = y_s_2_plus_3bf - y_s_2bf
            self.y_d_3bf = self.y_d_2_plus_3bf - self.y_d_2bf
