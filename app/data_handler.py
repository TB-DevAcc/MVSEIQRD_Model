import pandas as pd
import numpy as np
import geopands as gpd


class DataHandler:
    """
    Read data from files into program
    """
    def __init__(
        self,
        default_age_group_data_path="data/simulation_test/altersgruppen.csv",
        default_hospital_beds_data_path="data/simulation_test/krankenhausbetten.csv",
        default_recorded_covid_cases_path="data/RKI_COVID19.csv",
        default_district_geometry_path="data/RKI_Corona_Landkreise.shp"
    ):
        self.default_initial_data = self._load_simulation_initial_values(default_age_group_data_path, default_hospital_beds_data_path)
        self.default_recorded_cases = self._load_recorded_covid_cases(default_recorded_covid_cases_path)
        self.default_district_geometries = self._load_district_geometries(default_district_geometry_path)

    def _load_simulation_initial_values(self, default_age_group_data_path, default_hospital_beds_data_path) -> pd.DataFrame:
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

    def get_simulation_initial_values(self) -> dict:
        """
        Return loaded initial data for simulation

        Returns
        -------
        dict
            initial_values of districts for simulation
        """
        initial_values = self.default_initial_data.loc[:, self.default_initial_data.columns != "Landkreis"] \
            .set_index('IdLandkreis') \
            .to_dict('index')

        for (key, value) in initial_values.items():
            initial_values[key]['N_total'] = initial_values[key].pop('Insgesamt')
            initial_values[key]['B'] = initial_values[key].pop('Krankenhausbetten')
            initial_values[key]['N'] = np.array([value2 for key2, value2 in value.items() if key2 not in ['N', 'B']])
            initial_values[key] = {key2: initial_values[key][key2] for key2 in initial_values[key] if key2 in ['N_total', 'N', 'B']}

        return initial_values

    def _load_recorded_covid_cases(self, default_recorded_covid_cases_path) -> pd.DataFrame:
        """
        Read the covid csv-file into a DataFrame

        Parameters
        ----------
        default_recorded_covid_cases_path : str
            Path to the csv-File

        Returns
        -------
        pd.DataFrame
            A DataFrame of covid-19 from RKI-Germany
        """
        covid_df = pd.read_csv(default_recorded_covid_cases_path, sep=",", dtype={"IdLandkreis": str})
        covid_df = covid_df[["Bundesland", "Landkreis", "Altersgruppe", "Meldedatum", "AnzahlFall",
                             "AnzahlTodesfall", "NeuGenesen", "IdLandkreis"]]
        covid_df.rename(columns={"IdLandkreis": "RS"}, inplace=True)
        return covid_df

    def get_recorded_covid_cases(self):
        # TODO
        ...

    def _load_district_geometries(self, default_district_geometry_path) -> pd.DataFrame:
        """
        Read the covid Shapefile with all geometry data into a DataFrame

        Parameters
        ----------
        default_district_geometry_path : str
            Path to the shape file with the region-key (Regionalschlüssel) and the geometry-Data

        Returns
        -------
        pd.DataFrame
            A DataFrame of the Shapefile from RKI-Germany
        """
        rs_df = gpd.read_file(default_district_geometry_path)
        rs_df = rs_df[["GEN", "RS", "geometry"]]
        rs_df.sort_values(by=["RS"], inplace=True)
        return rs_df

    def get_district_geometries(self):
        # TODO
        ...

    def get_real_covid_data(self) -> pd.DataFrame:
        """

        Returns
        -------
        pd.DataFrame
        """
        c_plt_df = self.add_seven_day_average(self.default_recorded_cases, "AnzahlFall")
        c_plt_df = self.add_seven_day_average(c_plt_df, "NeuGenesen")
        c_plt_df = self.add_seven_day_average(c_plt_df, "AnzahlTodesfall")

        c_plt_df['Meldedatum'] = pd.to_datetime(c_plt_df['Meldedatum'])
        c_plt_df["NeuGenesen"] = c_plt_df["NeuGenesen"].abs()
        c_plt_df["NeuGenesen_seven_day_average"] = c_plt_df["NeuGenesen_seven_day_average"].abs()

        return c_plt_df

    def get_dates_of_covid_data(self) -> list:
        """
        Give a list of all dates

        Returns
        -------
        list
            a sorted list of all dates
        """
        lst = self.default_recorded_cases.drop_duplicates(subset="Meldedatum")
        lst = lst["Meldedatum"].tolist()
        lst = sorted(lst)
        return lst

    def get_values_of_day(self, day):
        return self.default_recorded_cases[(self.default_recorded_cases.Meldedatum == day)]

    def get_grouped_by_age(self, group_age):
        return self.default_recorded_cases[(self.default_recorded_cases.Altersgruppe == group_age)].sort_values(by=["Meldedatum"], inplace=True)

    def get_highest_infection(self, date, case) -> int:
        """
        Find the biggest number of all infections

        Parameters
        ----------
        date : list
            list of all dates
        case :

        Returns
        -------
        int
            Highest infection
        """
        new_df = self.default_recorded_cases[self.default_recorded_cases.Meldedatum.isin(date)]
        new_df = new_df.groupby(by=["Meldedatum", "RS"]).sum()

        # Ausreißer rausfiltern
        lst = new_df[case].tolist()
        if len(lst) > 100:
            lst = sorted(lst)
            del lst[len(lst) - 10:len(lst)]
        return max(lst)