import geopandas as gpd
import os.path
import numpy as np
import pandas as pd
from typing import Tuple


class DataHandler:
    """
    Read data from files into program
    """

    def __init__(
        self,
        default_age_group_data_path="data/simulation_test/altersgruppen.csv",
        default_hospital_beds_data_path="data/simulation_test/krankenhausbetten.csv",
        default_recorded_covid_cases_path="data/RKI_COVID19.csv",
        default_districts_geometry_path="data/RKI_Corona_Landkreise.geojson",
    ):
        self.default_initial_data = self._load_simulation_initial_values(
            default_age_group_data_path, default_hospital_beds_data_path
        )
        if os.path.isfile(default_recorded_covid_cases_path):
            self.recorded_cases = self._load_recorded_covid_cases(default_recorded_covid_cases_path)
        else:
            print(f"No file at {default_recorded_covid_cases_path}! Loading recorded cases was skipped.")

        if os.path.isfile(default_districts_geometry_path):
            self.district_geometries = self._load_district_geometries(default_districts_geometry_path)
        else:
            print(f"No file at {default_districts_geometry_path}! Loading geometry data was skipped.")

    def _load_simulation_initial_values(
        self, age_group_data_path, hospital_beds_data_path
    ) -> pd.DataFrame:
        """
        Load base data of districts for simulation of csv files

        Parameters
        ----------
        age_group_data_path : str
            path to csv file containing base data of age groups
        hospital_beds_data_path : str
            path to csv file containing base data of hospital beds

        Returns
        -------
        pd.DataFrame
            DataFrame with complete base data of districts for simulation
        """
        self.age_groups = self._prepare_age_groups(age_group_data_path)
        self.beds = self._prepare_hospital_beds(hospital_beds_data_path)

        base_data = pd.merge(self.age_groups, self.beds, on="IdLandkreis", how="left")

        base_data["Krankenhausbetten"] = (
            base_data["Krankenhausbetten"] / 100000 * base_data["Insgesamt"]
        )
        # Fix base data for one district manually
        base_data.loc[base_data["IdLandkreis"] == 3159, "Krankenhausbetten"] = (
            125.8 / 100000 * base_data.loc[base_data["IdLandkreis"] == 3159, "Insgesamt"]
        )
        base_data["Krankenhausbetten"] = (
            base_data["Krankenhausbetten"].round(decimals=0).astype("int32")
        )

        return base_data

    def _prepare_age_groups(self, age_groups_path) -> pd.DataFrame:
        """
        Prepares base data of districts for use in simulation

        Parameters
        ----------
        age_groups_path : str
            path to csv file containing base data

        Returns
        -------
        pd.DataFrame
            Dataframe with prepared values with base data of districts
        """
        age_groups = pd.read_csv(
            age_groups_path,
            delimiter=";",
            skiprows=5,
            usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,],
        )
        age_groups = age_groups[:-4]
        age_groups = age_groups.rename(
            columns={"Unnamed: 1": "IdLandkreis", "Unnamed: 2": "Landkreis"}
        )

        age_groups.loc[:, age_groups.columns != "Landkreis"] = (
            age_groups.loc[:, age_groups.columns != "Landkreis"].replace("-", "0").astype("int32")
        )

        age_groups["0 bis 4 Jahre"] = age_groups[["unter 3 Jahre", "3 bis unter 6 Jahre"]].sum(
            axis=1
        )
        age_groups["5 bis 14 Jahre"] = age_groups[
            ["6 bis unter 10 Jahre", "10 bis unter 15 Jahre"]
        ].sum(axis=1)
        age_groups["15 bis 34 Jahre"] = age_groups[
            [
                "15 bis unter 18 Jahre",
                "18 bis unter 20 Jahre",
                "20 bis unter 25 Jahre",
                "25 bis unter 30 Jahre",
                "30 bis unter 35 Jahre",
            ]
        ].sum(axis=1)
        age_groups["35 bis 59 Jahre"] = age_groups[
            [
                "35 bis unter 40 Jahre",
                "40 bis unter 45 Jahre",
                "45 bis unter 50 Jahre",
                "50 bis unter 55 Jahre",
                "55 bis unter 60 Jahre",
            ]
        ].sum(axis=1)
        age_groups["60 bis 79 Jahre"] = (
            age_groups[["60 bis unter 65 Jahre", "65 bis unter 75 Jahre"]].sum(axis=1)
            + 0.047 * age_groups["Insgesamt"]
        )
        age_groups["80 Jahre und älter"] = (
            age_groups[["75 Jahre und mehr"]].sum(axis=1) - 0.047 * age_groups["Insgesamt"]
        )

        age_groups = age_groups.iloc[:, [0, 1, -6, -5, -4, -3, -2, -1]]
        age_groups = age_groups.assign(
            Insgesamt=age_groups.iloc[:, [-6, -5, -4, -3, -2, -1]].sum(axis=1)
        )
        age_groups.loc[:, age_groups.columns != "Landkreis"] = age_groups.loc[
            :, age_groups.columns != "Landkreis"
        ].astype("int32")
        age_groups = age_groups[~age_groups["Landkreis"].str.contains("\(bis")]
        age_groups = age_groups[~age_groups["Landkreis"].str.contains("\(b.")]

        return age_groups

    def _prepare_hospital_beds(self, hospital_beds_path) -> pd.DataFrame:
        """
        Prepares base data of districts for use in simulation

        Parameters
        ----------
        hospital_beds_path : str
            path to csv file containing base data

        Returns
        -------
        pd.DataFrame
            Dataframe with prepared values with base data of districts
        """
        beds = pd.read_csv(hospital_beds_path, delimiter=";", usecols=[0, 2])

        beds = beds.rename(
            columns={"krs1214": "IdLandkreis", "krankenhausbetten2014": "Krankenhausbetten",}
        )

        beds["IdLandkreis"] = beds["IdLandkreis"] / 1000
        beds.loc[:, "IdLandkreis"] = beds.loc[:, "IdLandkreis"].astype("int32")

        beds["Krankenhausbetten"] = pd.to_numeric(
            beds["Krankenhausbetten"].str.replace(",", "."), errors="coerce"
        )

        return beds

    def get_simulation_initial_values(self, number_of_districts: int = 401) -> dict:
        """
        Return loaded initial data for simulation

        Parameters
        ----------
        number_of_districts : int
            Specify the number of districts the result should have (Allowed: 1, 16, 401)

        Returns
        -------
        dict
            initial_values of districts for simulation
        """
        initial_values = (
            self.default_initial_data.loc[:, self.default_initial_data.columns != "Landkreis"]
            .set_index("IdLandkreis")
            .to_dict("index")
        )

        for district, district_val in initial_values.items():
            initial_values[district]["N_total"] = initial_values[district].pop("Insgesamt")
            initial_values[district]["B"] = initial_values[district].pop("Krankenhausbetten")
            initial_values[district]["N"] = np.array(
                [
                    value
                    for age_group, value in district_val.items()
                    if age_group not in ["N_total", "B"]
                ]
            )
            initial_values[district] = {
                key: initial_values[district][key]
                for key in initial_values[district]
                if key in ["N_total", "N", "B"]
            }

        if number_of_districts == 1:
            return {
                    0: {
                        'N_total': np.sum([value['N_total'] for value in initial_values.values()]),
                        'B': np.sum([value['B'] for value in initial_values.values()]),
                        'N': np.sum([value['N'] for value in initial_values.values()], axis=(0))
                    }
                }
        elif number_of_districts == 16:
            return {
                    i: {
                        'N_total': np.sum([value['N_total'] for key, value in initial_values.items() if i * 1000 <= int(key) < (i + 1) * 1000]),
                        'B': np.sum([value['B'] for key, value in initial_values.items() if i * 1000 <= int(key) < (i + 1) * 1000]),
                        'N': np.sum([value['N'] for key, value in initial_values.items() if i * 1000 <= int(key) < (i + 1) * 1000], axis=(0))
                    }
                    for i in range(1, 16+1)
                }
        else:
            return initial_values

    def real_seir(self, df, vac_df):
        """
        Return dictionary as real seir model

        Parameters
        ----------
        df : df as covid_df

        Returns
        -------
        dict : parameters of real seir
        """



        c_df = df[["Meldedatum", "AnzahlFall", "AnzahlTodesfall", "NeuGenesen"]]
        c_df = c_df.groupby("Meldedatum").sum().abs().reset_index()
        c_df['Meldedatum'] = pd.to_datetime(c_df['Meldedatum'], format="%Y/%m/%d")
        vac_df.rename(columns={"Impfdatum": "Meldedatum"}, inplace=True)
        vac_df.rename(columns={"Anzahl": "AnzahlImpfungen"}, inplace=True)
        indexNames1 = vac_df[vac_df['Impfschutz'] == 1].index
        indexNames3 = vac_df[vac_df['Impfschutz'] == 3].index
        # Delete these rows indexes from dataFrame
        vac_df.drop(indexNames1, inplace=True)
        vac_df.drop(indexNames3, inplace=True)

        vac_df = vac_df[["Meldedatum", "AnzahlImpfungen"]]
        vac_df = vac_df.groupby(by="Meldedatum").sum().reset_index()
        vac_df['Meldedatum'] = pd.to_datetime(vac_df['Meldedatum'], format="%Y/%m/%d")
        # merge covid_df and vac_df
        sird = pd.merge(
            left=c_df,
            right=vac_df,
            on="Meldedatum",
            how="outer"
        ).fillna(0)

        sum_sird = sird.sum(axis=1)
        N = 83240000
        s = []
        date = c_df["Meldedatum"].tolist()
        for i in sum_sird:
            s += [N - i]
            N -= i
        s_df = pd.DataFrame(list(zip(date, s)), columns=['Meldedatum', 'AnzahlAnfälligen'])
        # merge with S = susceptible
        sird = pd.merge(
            left=sird,
            right=s_df,
            on="Meldedatum",
            how="outer"
        ).fillna(0)
        params = sird.copy()
        params = params.set_index("Meldedatum")
        param = {}
        for key, values in params.items():
            param[key] = values

        return param












    def _load_recorded_covid_cases(self, recorded_covid_cases_path) -> pd.DataFrame:
        """
        Read the covid csv-file into a DataFrame - does the same as 'read_covid_csvfile'

        Parameters
        ----------
        recorded_covid_cases_path : str
            Path to the csv-File

        Returns
        -------
        pd.DataFrame
            A DataFrame of covid-19 from RKI-Germany
        """
        covid_df = pd.read_csv(recorded_covid_cases_path, sep=",", dtype={"IdLandkreis": str})
        covid_df = covid_df[
            [
                "Bundesland",
                "Landkreis",
                "Altersgruppe",
                "Meldedatum",
                "AnzahlFall",
                "AnzahlTodesfall",
                "NeuGenesen",
                "IdLandkreis",
            ]
        ]
        covid_df.rename(columns={"IdLandkreis": "RS"}, inplace=True)
        return covid_df

    def get_recorded_covid_cases(self, recorded_covid_cases_path=None) -> pd.DataFrame:
        """
        Recorded covid cases of germany from RKI, load by default

        Parameters
        ----------
        recorded_covid_cases_path : str
            Path to RKI Covid file

        Returns
        -------
        pd.DataFrame
            Recorded covid cases of germany from RKI
        """
        if recorded_covid_cases_path is None:
            return self.recorded_cases
        else:
            return self._load_recorded_covid_cases(recorded_covid_cases_path)

    def _load_district_geometries(self, districts_geometry_path) -> pd.DataFrame:
        """
        Read the covid Shapefile with all geometry data into a DataFrame - does the same as 'read_geometryfile'

        Parameters
        ----------
        districts_geometry_path : str
            Path to the shape file with the region-key (Regionalschlüssel) and the geometry-Data

        Returns
        -------
        pd.DataFrame
            A DataFrame of the Shapefile from RKI-Germany
        """
        rs_df = gpd.read_file(districts_geometry_path)
        rs_df = rs_df[["GEN", "RS", "geometry"]]
        rs_df.sort_values(by=["RS"], inplace=True)
        return rs_df

    def get_district_geometries(self, districts_geometry_path=None) -> pd.DataFrame:
        """
        Geometry data of districts in germany

        Parameters
        ----------
        districts_geometry_path : str
            Path to file with geometry shapes

        Returns
        -------
        pd.DataFrame
            Geometry shapes
        """
        if districts_geometry_path is None:
            return self.district_geometries
        else:
            return self._load_district_geometries(districts_geometry_path)

    def prepare_real_covid_data(self, covid_data: pd.DataFrame = None) -> pd.DataFrame:
        """

        Parameters
        ----------
        covid_data : pd.DataFrame
            pd.DataFrame to use as a base for preparation

        Returns
        -------
        pd.DataFrame
        """
        if covid_data is None:
            covid_data = self.recorded_cases

        c_plt_df = self._add_seven_day_average(covid_data, "AnzahlFall")
        c_plt_df = self._add_seven_day_average(c_plt_df, "NeuGenesen")
        c_plt_df = self._add_seven_day_average(c_plt_df, "AnzahlTodesfall")

        c_plt_df["Meldedatum"] = pd.to_datetime(c_plt_df["Meldedatum"])
        c_plt_df["NeuGenesen"] = c_plt_df["NeuGenesen"].abs()
        c_plt_df["NeuGenesen_seven_day_average"] = c_plt_df["NeuGenesen_seven_day_average"].abs()

        return c_plt_df

    def prepare_simulated_covid_data(self, model, covid_data: dict, mode: str = "I", start_date: str = "2020-01-01") -> Tuple:
        """
        Creates a dataframe for usage in MyFuncAnimator from simulation results

        Parameters
        ----------
        model
        start_date : str
            date animation should start - default: 2020-01-01
        covid_data : dict
            simulation results
        mode : str
            Mode which class should be viewed - I, E or V

        Returns
        -------
        Tuple
            Dataframe with prepared data, geo data and dates of dataframe

        """
        sim_type = model.simulator.simulation_type
        select_classes = []
        if mode == "I":
            if "I3" in sim_type:
                select_classes = ["I_asym", "I_sym", "I_sev"]
            elif "I" in sim_type:
                select_classes = ["I"]
        elif mode == "E":
            if "E2" in sim_type:
                select_classes = ["E_tr", "E_nt"]
            if "E" in sim_type:
                select_classes = ["E"]
        elif mode == "V" and "V" in sim_type:
            select_classes = ["V"]
        else:
            print("Given mode is not correct, using fallback...")
            select_classes = ["I"]

        data = np.sum([covid_data[classes] for classes in select_classes], axis=(0))
        data_frame = pd.DataFrame(np.sum(data, axis=(2)))

        map_params = model.controller.map_params
        data_frame.rename(columns=map_params, inplace=True)
        data_frame = data_frame.assign(
            Meldedatum=pd.date_range(start_date, periods=covid_data[select_classes[0]].shape[0])
        )
        data_frame = pd.melt(
            data_frame, id_vars=["Meldedatum"], value_vars=map_params.values()
        )
        data_frame.rename(
            columns={"variable": "RS", "value": "AnzahlFall"}, inplace=True
        )
        data_frame["Meldedatum"] = data_frame["Meldedatum"].astype("str")
        data_frame = data_frame.assign(Bundesland="")
        # disable two districts causing problems with geopandas
        data_frame = data_frame[~data_frame["RS"].isin(['11000', '16056'])]

        # len(map_params) == 1 means that it is one district "Germany" with RS "0"
        dates = None
        if len(map_params) > 1:
            dates = data_frame[data_frame["RS"] == "01001"]["Meldedatum"]
        else:
            dates = data_frame[data_frame["RS"] == "0"]["Meldedatum"]

        geo = self.get_district_geometries()
        if len(map_params) > 1:
            geo = pd.concat(
                [geo[geo["RS"] == district] for district in map_params.values()]
            )

        return data_frame, geo, dates

    def _add_seven_day_average(
        self, covid_data: pd.DataFrame, column_name: str
    ) -> pd.DataFrame:
        """
        Adds new column to given dataFrame for saving a seven day average

        Parameters
        ----------
        covid_data : pd.DataFrame
            DataFrame with original data
        column_name
            Name of column to add

        Returns
        -------
        pd.DataFrame
            DataFrame with added column

        """
        new_df = covid_data.groupby(by="Meldedatum").sum()
        name = column_name + "_" + "seven_day_average"
        new_df[name] = new_df[column_name].rolling(window=7).mean()
        return new_df.reset_index()

    def get_dates_of_covid_data(self, covid_data: pd.DataFrame = None) -> list:
        """
        Extract a list of all dates in given DataFrame

        Parameters
        ----------
        covid_data : pd.DataFrame
            pd.DataFrame to use as a base for preparation

        Returns
        -------
        list
            a sorted list of all dates
        """
        if covid_data is None:
            covid_data = self.recorded_cases

        lst = covid_data.drop_duplicates(subset="Meldedatum")
        lst = lst["Meldedatum"].tolist()
        lst = sorted(lst)
        return lst

    def get_values_of_day(self, day, covid_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Get Values of given day

        Parameters
        ----------
        covid_data : pd.DataFrame
            pd.DataFrame to use as a base
        day

        Returns
        -------
        pd.DataFrame
            pd.DataFrame with data of given day
        """
        if covid_data is None:
            covid_data = self.recorded_cases

        return covid_data[(covid_data.Meldedatum == day)]

    def get_grouped_by_age(self, group_age, covid_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Group by given age

        Parameters
        ----------
        covid_data : pd.DataFrame
            pd.DataFrame to use as a base
        group_age

        Returns
        -------
        pd.DataFrame
            pd.DataFrame grouped by given age
        """
        if covid_data is None:
            covid_data = self.recorded_cases

        return covid_data[covid_data.Altersgruppe == group_age].sort_values(by=["Meldedatum"])

    def get_highest_infection(
        self, date, filter_name: str, covid_data: pd.DataFrame = None
    ) -> int:
        """
        Find the biggest number of all infections

        Parameters
        ----------
        covid_data : pd.DataFrame
            pd.DataFrame to use as a base
        date : list
            list of all dates
        filter_name : str
            name of column where should be looked at

        Returns
        -------
        int
            Number of highest infection
        """
        if covid_data is None:
            covid_data = self.recorded_cases

        new_df = (
            covid_data[covid_data.Meldedatum.isin(date)].groupby(by=["Meldedatum", "RS"]).sum()
        )

        # Ausreißer rausfiltern
        lst = new_df[filter_name].tolist()
        if len(lst) > 100:
            lst = sorted(lst)
            del lst[len(lst) - 10 : len(lst)]
        return max(lst)
