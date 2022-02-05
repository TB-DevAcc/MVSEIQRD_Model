import geopandas as gpd
import pandas as pd

class Covid_Data_Handler:
    

    """
        Read the covid csv-file into a DataFrame
    
        input:
        csv_file - (String) Path to the csv-File
    
    
        return:
        covid_df - (DataFrame) A DataFrame of covid-19 from RKI-Germany
    
    """
    
    def read_covid_csvfile(self, csv_file):
        covid_df = pd.read_csv(csv_file, sep=",", dtype={"IdLandkreis": str})
        covid_df= covid_df[["Bundesland","Landkreis","Altersgruppe", "Meldedatum", "AnzahlFall","AnzahlTodesfall","NeuGenesen","IdLandkreis"]] 
        covid_df.rename(columns ={"IdLandkreis":"RS"},inplace=True)
        self.covid_df = covid_df
        return self.get_covid_df()
    
    """
        Read the covid Shapefile with all geometry data into a DataFrame
        
        input:
        rs_file - Path to the shape file with the region-key (Regionalschlüssel) and the geometry-Data
    
        return:
        rs_df - (DataFrame) A DataFrame of the Shapefile from RKI-Germany
    
    """
    
    def read_geometryfile(self, rs_file="/data/RKI_Corona_Landkreise.shp"):        
        rs_df = gpd.read_file(rs_file)
        rs_df= rs_df[["GEN","RS", "geometry"]]
        rs_df.sort_values(by=["RS"], inplace=True)
        self.rs_df = rs_df
        return self.get_rs_df()
    

    def grouped_age(self, c_df, group_age):
        c_df = c_df[(c_df.Altersgruppe == group_age)]
        c_df.sort_values(by=["Meldedatum"], inplace=True)
        return c_df

    """
        Extract the covid-Data in state and/or group_age 
        
        input:
        c_df      - (DataFrame) covid DataFrame
        rs_df     - (DataFrame) geometry-Data (eingelesenes Shapefile)
        state     - (String) "Bayern","Baden-Württemberg", usw...
        group_age - (String) default ="all", input as group_age alias "A05-A14"

        Return:
        vs_df - (DataFrame) DataFrame with a grouped state and ages
        rs    - (DataFrame DataFrame with geometry Data with state and ages

    """



    def values_today_germany(self, c_df, day):
        vtg_df = c_df[(c_df.Meldedatum == day)]
        return vtg_df

    """
        Give a list of all dates 
        
        input:
        c_df  - (DataFrame) covid DataFrame
        
        return:
        lst - (list) a sorted list of all dates

    """

    def list_of_dates(self, c_df):
        lst = c_df.drop_duplicates(subset="Meldedatum")
        lst = lst["Meldedatum"].tolist()
        lst = sorted(lst)
        return lst

    """
        Find the biggest number of all infections
        
        input:
        c_df  - (DataFrame) covid DataFrame
        date  - (list) list of all dates
        
        return:
        
    """

    def maxNum(self, c_df, date, fall):
        new_df = c_df[c_df.Meldedatum.isin(date)]
        new_df = new_df.groupby(by=["Meldedatum", "RS"]).sum()
        
        #Ausreißer rausfiltern 
        lst = new_df[fall].tolist()
        if len(lst) > 100:
            lst = sorted(lst)
            del lst[len(lst) - 10:len(lst)]
        max_Num = max(lst)
        return max_Num

    
    """
        Give the seven day average from the DataFrame to the DataFrame
        
        input:
        c_df  - (DataFrame) covid DataFrame
        fall  - (String) is the infections or deaths ...
        
        return:
        new_df - (DataFrame) with a new column with seven day average
    
    """
    def add_seven_day_average(self,c_df,fall=None):
        new_df = c_df.groupby(by="Meldedatum").sum()
        name = fall+"_"+"seven_day_average"
        new_df[name] = new_df[fall].rolling(window=7).mean()
        return new_df.reset_index()
    
    
    
    
    def calc_real_data_plt(self,c_df=None):

        c_plt_df = self.add_seven_day_average(c_df,"AnzahlFall")
        c_plt_df = self.add_seven_day_average(c_plt_df,"NeuGenesen")
        c_plt_df = self.add_seven_day_average(c_plt_df,"AnzahlTodesfall")

        c_plt_df['Meldedatum'] = pd.to_datetime(c_plt_df['Meldedatum'])
        c_plt_df["NeuGenesen"] = c_plt_df["NeuGenesen"].abs()
        c_plt_df["NeuGenesen_seven_day_average"] = c_plt_df["NeuGenesen_seven_day_average"].abs()
        self.c_plt_df = c_plt_df
        
        return self.get_real_data_plt()
    
    def get_real_data_plt(self):
        return self.c_plt_df
    
    def get_covid_df(self):
        return self.covid_df
    
    def get_rs_df(self):
        return self.rs_df
    
    
    