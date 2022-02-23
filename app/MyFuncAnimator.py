#import matplotlib.colors as colors
from tqdm.notebook import *
from app.data_handler import DataHandler
import numpy as np
import geopandas as gpd
import pandas as pd
from matplotlib.colors import Normalize, LogNorm
import matplotlib.pyplot as plt
import matplotlib as mpl 


class MyFuncAnimator:
    
    """
        This is the init Method 
        df       - (DataFrame) covid DataFrame
        date_lst - (list) list of dates
        fall     - (String) the infections, deaths ...
    
    """
    
    def __init__(self, df, rs, date_lst, fall="AnzahlFall"):
        self.dh = DataHandler()
        self.fig, self.ax = plt.subplots()
        self.length = len(date_lst)
        self.df = df
        self.rs = rs
        self.date_lst = date_lst
        self.pbar = tqdm(total=len(date_lst) + 1)
        self.maxNum = self.dh.get_highest_infection(date_lst, fall, df)
        self.fall = fall
        self.is_germany = self.is_germany(df)
        self.top_cities = {
            "Berlin": (13.404954, 52.520008), 
            "Köln": (6.953101, 50.935173),
            "Düsseldorf": (6.782048, 51.227144),
            "Frankfurt am Main": (8.682127, 50.110924),
            "Hamburg": (9.993682, 53.551086),
            "Leipzig": (12.387772, 51.343479),
            "München": (11.576124, 48.137154),
            "Dortmund": (7.468554, 51.513400),
            "Stuttgart": (9.181332, 48.777128),
            "Nürnberg": (11.077438, 49.449820),
            "Hannover": (9.73322, 52.37052)
            }
        
    """
        Method that finds out whether the handed over DataFrame is all of Germany or not
        
        input:
        df         - (DataFrame) covid DataFrame
        
        return:
        is_germany - (boolean) is it all germany then true, else false
        
    """
    
    def is_germany(self,df):
        g_df = df.drop_duplicates(subset=["Bundesland"])
        if len(g_df.index) == 16:
            is_germany = True
        else:
            is_germany = False
        return is_germany
    
    """
        It returns the next DataFrame on the given date with merged rs DataFrame
        
        input: 
        df         - (DataFrame) covid DataFrame
        rs         - (DataFrame) rs DataFrame
        date       - (String)   date
        
        return:
        new_df  - (DataFrame) merged DataFrame with geometry file for the animation 
    
    """

    def next_values(self, df, rs, date):
        new_df = self.dh.get_values_of_day(date, df)
        new_df = new_df.groupby(by="RS").sum().groupby(level=[0]).cumsum()

        new_df = pd.merge(
            left=new_df,
            right=rs,
            on="RS",
            how="outer"
        )
        new_df = new_df.replace(np.nan, int(0))
        new_df.sort_values(by=["RS"], inplace=True)
        lst2 = new_df[self.fall].tolist()

        new_df = rs.assign(Anzahl=lst2)
        return new_df

    
    
    def animate(self, n):
        self.ax.clear()
        self.pbar.update(1)
        self.ax.set_title("New cases of COVID-19 on {}".format(self.date_lst[n].split()[0]), fontdict={"fontsize": "25", "fontweight" : "3"})
        self.ax.set_facecolor("lightblue")
        
        if self.is_germany == True:
            for c in self.top_cities.keys():
                # Plot city name.
                self.ax.text(
                    x=self.top_cities[c][0], 
                    # Add small shift to avoid overlap with point.
                    y=self.top_cities[c][1] + 0.08, 
                    s=c, 
                    fontsize=12,
                    ha="center", 
                    )
                # Plot city location centroid.
                self.ax.plot(
                    self.top_cities[c][0], 
                    self.top_cities[c][1], 
                marker="o",
                c="black", 
                alpha=0.5
                )
        
        
        
        
        self.t_df = self.next_values(self.df, self.rs, self.date_lst[n])
        self.t_df.plot(ax=self.ax,
                  column="Anzahl",
                  #vmin=0, vmax=self.maxNum,
                  norm = self.norm,
                  categorical=False,
                  legend=False,
                  cmap="brg_r"
                  )
        
        return self.t_df,
    
    """
    Animate the DataFrame
    """

    def FuncAnimator(self):
        self.ax.set_axis_off()
        self.norm = Normalize(vmin=1, vmax=self.maxNum)
        cax = self.fig.add_axes([self.ax.get_position().x0,self.ax.get_position().y0,self.ax.get_position().width, 0.02])
        sm = plt.cm.ScalarMappable(cmap="brg_r", norm=self.norm)
        cb = self.fig.colorbar(sm, cax=cax, orientation="horizontal")
        
        ani = mpl.animation.FuncAnimation(self.fig, self.animate, frames=self.length, interval=self.length,
                                          repeat=False, blit=False)
        # ani.save('myAnimation.gif', writer='imagemagick', fps=2)
        return ani
    
    
    
    