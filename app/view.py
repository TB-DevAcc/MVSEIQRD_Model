import plotly.graph_objects as go
import plotly.express as px
import plotly
import numpy as np


class View:

    def __init__(self, real_area_data: dict = None, real_params: dict = None, sim_area_data: dict = None,
        sim_params: dict = None, t=None):
        self.real_area_data = real_area_data
        self.sim_area_data = sim_area_data
        self.real_params = real_params
        self.sim_params = sim_params
        self.t = t

    def plot_sim(self, params: dict = None, sim: bool = True, seir: bool = True, t=None):

        """
        Plotting the simulated data

        Parameters
        ----------
        params : dict
        sim : boolean
        seir: bool
        t : np.array

        """

        # check if params contain values and update or use self.params otherwise
        if not params:
            if not self.params:
                raise ValueError()
        else:
            self.sim_params = params

        if seir:
            self.seir_plot(self.sim_params, t)

        else:
            self.seiqrds_plot(self.sim_params, t)

    def seir_plot(self, params, t):

        """
        Plotting the "S I" or S E I R"

        Parameters
        ----------
        params : dict
        t : np.array

        Returns
        -------
        fig plotly plot

        """

        fig = go.Figure()

        for key, value in params.items():
            fig.add_trace(go.Line(x=t, y=np.sum(value, axis=1),
                                  mode='lines',
                                  name=key))
            fig.update_layout(
                autosize=False,
                width=700,
                height=700)

        return fig.show()

    def seiqrds_plot(self, params, t):

        """
        Plotting the "M V S E2 I3 Q3 R D"

        Parameters
        ----------
        params : dict
        t : np.array

        Returns
        -------
        fig plotly plot

        """

        fig = go.Figure()

        for key, value in params.items():
            fig.add_trace(go.Line(x=t, y=np.sum(value, axis=(1, 2)),
                                  mode='lines',
                                  name=key))
            fig.update_layout(
                autosize=False,
                width=700,
                height=700)

        return fig.show()

    def plot_real_data(self, df):

        """
        Plotting the real covid data

        Parameters
        ----------
        df : DataFrame

        Returns
        -------
        fig
            plotly plot
        """

        try:
            self.c_plt_df = df
            fig = go.Figure()

            fig.add_trace(go.Line(x=self.c_plt_df["Meldedatum"], y=self.c_plt_df["AnzahlFall"],
                                  mode='lines',
                                  name='Anzahl Fälle'))
            fig.add_trace(go.Line(x=self.c_plt_df["Meldedatum"], y=self.c_plt_df["AnzahlFall_seven_day_average"],
                                  mode='lines',
                                  name='Anzahl Fälle 7-Tage-Mittelwert'))
            fig.add_trace(go.Line(x=self.c_plt_df["Meldedatum"], y=self.c_plt_df["NeuGenesen"],
                                  mode='lines',
                                  name='Neu Genesen', visible="legendonly"))
            fig.add_trace(go.Line(x=self.c_plt_df["Meldedatum"], y=self.c_plt_df["NeuGenesen_seven_day_average"],
                                  mode='lines',
                                  name='Neu Genesen 7-Tage-Mittelwert', visible="legendonly"))
            fig.add_trace(go.Line(x=self.c_plt_df["Meldedatum"], y=self.c_plt_df["AnzahlTodesfall"],
                                  mode='lines',
                                  name="Anzahl der Todesfälle", visible="legendonly"))
            fig.add_trace(go.Line(x=self.c_plt_df["Meldedatum"], y=self.c_plt_df["AnzahlTodesfall_seven_day_average"],
                                  mode='lines',
                                  name="Anzahl der Todesfälle 7-Tage-Mittelwert", visible="legendonly"))
        except Exception as e:
            print(e)

        return fig.show()
