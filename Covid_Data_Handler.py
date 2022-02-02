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
        
        return covid_df
    
    """
        Read the covid Shapefile with all geometry data into a DataFrame
        
        input:
        rs_file - Path to the shape file with the region-key (Regionalschlüssel) and the geometry-Data
    
        return:
        rs_df - (DataFrame) A DataFrame of the Shapefile from RKI-Germany
    
    """
    
    def read_geometryfile(self, rs_file):        
        rs_df = gpd.read_file("RKI_Corona_Landkreise.shp")
        rs_df= rs_df[["GEN","RS", "geometry"]]
        rs_df.sort_values(by=["RS"], inplace=True)
        return rs_df

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

    def value_of_states(self, c_df, rs_df, state, group_age="all"):
        if group_age != "all":
            c_df = c_df[(c_df.Altersgruppe == group_age)]
        groups = c_df.groupby(c_df.Bundesland)
        vs_df = groups.get_group(state)

        duplicate = vs_df.drop_duplicates(subset=["RS"])
        dup_lst = duplicate["RS"].tolist()
        rs_df = rs_df[rs_df["RS"].isin(dup_lst)]
        return vs_df, rs_df
    
    """
        Extract the covid-Data in region and/or group_age 
        
        input:
        c_df      - (DataFrame) covid DataFrame
        rs_df     - (DataFrame) geometry-Data (eingelesenes Shapefile)
        region    - (String) "01001","01002"...
        group_age - (String) default ="all", input as group_age alias "A05-A14"

        Return:
        vs_df - (DataFrame) DataFrame with a grouped regioin and ages
        rs    - (DataFrame DataFrame with geometry Data with state and ages

    """

    def value_of_regions(self, c_df, rs_df, rs, group_age="all"):
        if group_age != "all":
            c_df = c_df[(c_df.Altersgruppe == group_age)]

        groups = c_df.groupby(c_df.RS)
        c_df = groups.get_group(rs)

        rs_df = rs_df[rs_df["RS"].isin([rs])]
        return c_df, rs_df

    """
        Get all rows with the input day
        
        input:
        c_df  - (DataFrame) covid DataFrame
        day   - (String)
        
        return:
        vtg_df - (DataFrame) extract all df with the input day
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
    def add_seven_day_average(self,c_df,fall="AnzahlFall"):
        new_df = c_df.groupby(by="Meldedatum").sum()
        new_df["seven-day-average"] = new_df[fall].rolling(window=7).mean()
        return new_df