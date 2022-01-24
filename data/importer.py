import pandas as pd
import json
import numpy as np


class Importer:
    def __init__(
            self,
            default_age_group_data_path='simulation_test/altersgruppen.csv',
            default_hospital_beds_data_path='simulation_test/krankenhausbetten.csv'
    ):
        self.default_base_data = self._load_base_values(default_age_group_data_path, default_hospital_beds_data_path)

    def _load_base_values(self, default_age_group_data_path, default_hospital_beds_data_path) -> pd.DataFrame:
        """
        Load base data of districts for simulation of csv files

        Parameters
        ----------
        default_age_group_data_path : str
            path to csv file containing base data of age groups
        default_hospital_beds_data_path : str
            path to csv file containing base data of hospital beds

        Returns
        -------
        pd.DataFrame
            DataFrame with complete base data of districts for simulation
        """
        self.age_groups = self._prepare_age_groups(default_age_group_data_path)
        self.beds = self._prepare_hospital_beds(default_hospital_beds_data_path)

        base_data = pd.merge(self.age_groups, self.beds, on="IdLandkreis", how="left")

        base_data["Krankenhausbetten"] = base_data["Krankenhausbetten"] / 100000 * base_data["Insgesamt"]
        # Fix base data for one district manually
        base_data.loc[base_data["IdLandkreis"] == 3159, "Krankenhausbetten"] \
            = 125.8 / 100000 * base_data.loc[base_data["IdLandkreis"] == 3159, "Insgesamt"]
        base_data["Krankenhausbetten"] = base_data["Krankenhausbetten"].round(decimals=0).astype('int32')

        return base_data

    def _prepare_age_groups(self, default_age_groups_path) -> pd.DataFrame:
        """
        Prepares base data of districts for use in simulation

        Parameters
        ----------
        default_age_groups_path : str
            path to csv file containing base data

        Returns
        -------
        pd.DataFrame
            Dataframe with prepared values with base data of districts
        """
        age_groups = pd.read_csv(default_age_groups_path, delimiter=";", skiprows=5,
                                 usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        age_groups = age_groups[:-4]
        age_groups = age_groups.rename(columns={"Unnamed: 1": "IdLandkreis", "Unnamed: 2": "Landkreis"})

        age_groups.loc[:, age_groups.columns != "Landkreis"] = \
            age_groups.loc[:, age_groups.columns != "Landkreis"].replace("-", "0").astype("int32")

        age_groups["0 bis 4 Jahre"] = age_groups[
            ["unter 3 Jahre",
             "3 bis unter 6 Jahre"]
        ].sum(axis=1)
        age_groups["5 bis 14 Jahre"] = age_groups[
            ["6 bis unter 10 Jahre",
             "10 bis unter 15 Jahre"]
        ].sum(axis=1)
        age_groups["15 bis 34 Jahre"] = age_groups[
            ["15 bis unter 18 Jahre",
             "18 bis unter 20 Jahre",
             "20 bis unter 25 Jahre",
             "25 bis unter 30 Jahre",
             "30 bis unter 35 Jahre"]
        ].sum(axis=1)
        age_groups["35 bis 59 Jahre"] = age_groups[
            ["35 bis unter 40 Jahre",
             "40 bis unter 45 Jahre",
             "45 bis unter 50 Jahre",
             "50 bis unter 55 Jahre",
             "55 bis unter 60 Jahre"]
        ].sum(axis=1)
        # TODO get a solution to divide the 75+ group from original data
        age_groups["60 bis 79 Jahre"] = age_groups[
            ["60 bis unter 65 Jahre",
             "65 bis unter 75 Jahre"]
        ].sum(axis=1) + 0.047 * age_groups["Insgesamt"]
        age_groups["80 Jahre und älter"] = age_groups[
            ["75 Jahre und mehr"]
        ].sum(axis=1) - 0.047 * age_groups["Insgesamt"]

        age_groups = age_groups.iloc[:, [0, 1, -6, -5, -4, -3, -2, -1]]
        age_groups = age_groups.assign(Insgesamt=age_groups.iloc[:, [-6, -5, -4, -3, -2, -1]].sum(axis=1))
        age_groups.loc[:, age_groups.columns != "Landkreis"] \
            = age_groups.loc[:, age_groups.columns != "Landkreis"].astype("int32")
        age_groups = age_groups[~age_groups["Landkreis"].str.contains("\(bis")]
        age_groups = age_groups[~age_groups["Landkreis"].str.contains("\(b.")]

        return age_groups

    def _prepare_hospital_beds(self, default_hospital_beds_path) -> pd.DataFrame:
        """
        Prepares base data of districts for use in simulation

        Parameters
        ----------
        default_hospital_beds_path : str
            path to csv file containing base data

        Returns
        -------
        pd.DataFrame
            Dataframe with prepared values with base data of districts
        """
        beds = pd.read_csv(default_hospital_beds_path, delimiter=";", usecols=[0, 2])

        beds = beds.rename(columns={"krs1214": "IdLandkreis", "krankenhausbetten2014": "Krankenhausbetten"})

        beds["IdLandkreis"] = beds["IdLandkreis"] / 1000
        beds.loc[:, "IdLandkreis"] = beds.loc[:, "IdLandkreis"].astype("int32")

        beds["Krankenhausbetten"] = pd.to_numeric(beds["Krankenhausbetten"].str.replace(',', '.'), errors='coerce')

        return beds

    def _to_dict(self):
        """
        Save loaded base data in json files for controller
        """
        # TODO implement
        tmp = self.default_base_data.loc[:, self.default_base_data.columns != "Landkreis"] \
            .set_index('IdLandkreis') \
            .to_dict('index')

        for (key, value) in tmp.items():
            tmp[key]['N_total'] = tmp[key].pop('Insgesamt')
            tmp[key]['B'] = tmp[key].pop('Krankenhausbetten')
            tmp[key]['N'] = np.array([value2 for key2, value2 in value.items() if key2 not in ['N', 'B']])

            # TODO: delete old keys
            [tmp[key].pop(key2) for key2, value2 in value.items() if key2 not in ['N', 'B']]

        return tmp
