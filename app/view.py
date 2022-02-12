import base64
import time
from pathlib import Path

import dash_bootstrap_components as dbc
import networkx as nx
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydot
from dash import dcc, html
from dash.dependencies import Input, Output, State
from jupyter_dash import JupyterDash
from pyvis.network import Network
from skimage import io


class View:
    def __init__(
        self,
        real_area_data: dict = None,
        real_params: dict = None,
        sim_area_data: dict = None,
        sim_params: dict = None,
        t=None,
    ):
        self.real_area_data = real_area_data
        self.sim_area_data = sim_area_data
        self.real_params = real_params
        self.sim_params = sim_params
        self.t = t
        self.network_svg_path = str(self._create_network_svg(return_b64=False))
        self.network_svg_b64 = (
            "data:image/svg+xml;base64," + str(self._create_network_svg(return_b64=True))[2:-1]
        )
        self.network_iframe_path = str(self._create_network_iframe())
        self.app = self._build_app()

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
            fig.add_trace(go.Line(x=t, y=np.sum(value, axis=1), mode="lines", name=key))
            fig.update_layout(autosize=False, width=700, height=700)

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
            fig.add_trace(go.Line(x=t, y=np.sum(value, axis=(1, 2)), mode="lines", name=key))
            fig.update_layout(autosize=False, width=700, height=700)

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

            fig.add_trace(
                go.Line(
                    x=self.c_plt_df["Meldedatum"],
                    y=self.c_plt_df["AnzahlFall"],
                    mode="lines",
                    name="Anzahl Fälle",
                )
            )
            fig.add_trace(
                go.Line(
                    x=self.c_plt_df["Meldedatum"],
                    y=self.c_plt_df["AnzahlFall_seven_day_average"],
                    mode="lines",
                    name="Anzahl Fälle 7-Tage-Mittelwert",
                )
            )
            fig.add_trace(
                go.Line(
                    x=self.c_plt_df["Meldedatum"],
                    y=self.c_plt_df["NeuGenesen"],
                    mode="lines",
                    name="Neu Genesen",
                    visible="legendonly",
                )
            )
            fig.add_trace(
                go.Line(
                    x=self.c_plt_df["Meldedatum"],
                    y=self.c_plt_df["NeuGenesen_seven_day_average"],
                    mode="lines",
                    name="Neu Genesen 7-Tage-Mittelwert",
                    visible="legendonly",
                )
            )
            fig.add_trace(
                go.Line(
                    x=self.c_plt_df["Meldedatum"],
                    y=self.c_plt_df["AnzahlTodesfall"],
                    mode="lines",
                    name="Anzahl der Todesfälle",
                    visible="legendonly",
                )
            )
            fig.add_trace(
                go.Line(
                    x=self.c_plt_df["Meldedatum"],
                    y=self.c_plt_df["AnzahlTodesfall_seven_day_average"],
                    mode="lines",
                    name="Anzahl der Todesfälle 7-Tage-Mittelwert",
                    visible="legendonly",
                )
            )
        except Exception as e:
            print(e)

        return fig.show()

    def _create_network_iframe(
        self,
        network_iframe_path=Path("assets/network.html"),
        dot_path=Path("data/param_graph.dot"),
    ):
        G = nx.DiGraph(nx.drawing.nx_pydot.read_dot(dot_path))
        net = Network(directed=True, notebook=True)
        net.from_nx(G)
        options = """
        var options = \
        {
            "nodes": {
                "font": {
                    "background": "rgba(255,125,104,0.77)"
                }
            },
            "edges": {
                "color": {
                    "inherit": true
                },
                "scaling": {
                    "max": 100
                },
                "font": {
                    "size": 9,
                    "background": "rgba(255,255,255,0.90)"
                },
                "smooth": {
                    "forceDirection": "none"
                }
            },
            "layout": {
                "hierarchical": {
                    "enabled": true,
                    "direction": "LR",
                    "sortMethod": "directed"
                }
            },
            "interaction": {
                "multiselect": true
            },
            "physics": {
                "hierarchicalRepulsion": {
                    "centralGravity": 0
                }
            }
        }
        """
        net.set_options(options)
        # net.show_buttons(filter_=True)
        # net.show(network_iframe_path)
        net.write_html(str(network_iframe_path), notebook=True)
        return network_iframe_path

    def _create_network_svg(
        self,
        network_svg_path=Path("assets/network.svg"),
        dot_path=Path("data/param_graph.dot"),
        return_b64=False,
    ) -> str:
        graphs = pydot.graph_from_dot_file(dot_path)
        graph = graphs[0]
        graph.set_bgcolor("transparent")
        graph.set_size(8)
        graph.write_svg(network_svg_path)
        if return_b64:
            return base64.b64encode(graph.create_svg())
        else:
            return network_svg_path

    def _build_app(self):
        """
        returns Jupyter-Dash Webapp
        """

        def build_fig(df=None):
            if not df:
                df = {}
            fig = px.line(df, title="SEIR Simulation classes over time")
            fig.update_layout(
                autosize=False,
                width=1200,
                height=500,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            return fig

        def build_slider(kwargs=None):
            """
            Build a slider from kwargs and default settings
            """
            sliderparams = {
                "min": 0,
                "max": 100,
                "value": 65,
                "step": 5.0,
                "marks": {
                    0: {"label": "0", "style": {"color": "#0b4f6c"}},
                    25: {"label": "25", "style": {"color": colors["text"]}},
                    50: {"label": "50", "style": {"color": colors["text"]}},
                    75: {"label": "75", "style": {"color": colors["text"]}},
                    100: {"label": "100", "style": {"color": "#f50"}},
                },
                "included": True,
                "disabled": False,  # Handles can't be moved if True
                # "vertical":True,
                # "verticalHeight":400,
            }
            if kwargs:
                for k, v in kwargs.items():
                    sliderparams[k] = v
                return dcc.Slider(**sliderparams)

        # Build App
        app = JupyterDash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],)

        colors = {
            "background": "#7a9e7e",
            "text": "#7FDBFF",
        }

        header = html.Div(
            className="row navbar navbar-dark bg-dark shadow-sm",
            id="header",
            children=[
                html.H1(
                    style={"text-align": "left", "color": colors["background"]},
                    children=["MVSEIQRD - Modell",],
                ),
            ],
        )

        slider_keys = [
            "sigma",
            "rho_mat",
            "rho_vac",
            "rho_rec",
            "nu",
            "beta_asym",
            "beta_sym",
            "beta_sev",
            "psi",
            "epsilon",
            "gamma_asym",
            "gamma_sym",
            "gamma_sev",
            "gamma_sev_r",
            "gamma_sev_d",
            "mu_sym",
            "mu_sev",
            "tau_asym",
            "tau_sym",
            "tau_sev",
        ]

        sliders = []
        for slider_key in slider_keys:
            slider_output_id = slider_key + "_output"
            slider_id = slider_key + "_slider"
            # Text label
            sliders.append(
                html.P(
                    children=slider_key,
                    className="mt-2 mb-0 ms-3",
                    style={"text-align": "left",},
                    id=slider_output_id,
                )
            )
            # Slider
            sliders.append(build_slider({"id": slider_id}))

        slider_col_1 = html.Div(
            className="col-2 my-auto align-middle",
            id="slider-col-1",
            children=sliders[: len(sliders) // 2],
        )
        slider_col_2 = html.Div(
            className="col-2 my-auto align-middle",
            id="slider-col-2",
            children=sliders[len(sliders) // 2 :],
        )

        button_col = html.Div(
            className="col-1 my-auto align-middle",
            style={"text-align": "center",},
            children=[
                dbc.Button(
                    id="loading-button",
                    className="my-auto",
                    n_clicks=0,
                    children=["Run Simulation"],
                ),
            ],
        )

        plot_col = html.Div(
            className="col-7 my-auto px-0 mx-0",
            id="sim-graph",
            children=[
                html.Div(
                    style={"text-align": "center",},
                    children=[
                        dcc.Loading(
                            children=[
                                html.Img(
                                    src=self.network_svg_b64,
                                    id="network-output",
                                    className="mx-auto mb-1 mt-5 pt-5",
                                    style={
                                        "width": 600,
                                        "height": 250,
                                        "text-align": "center",
                                        "background-color": "transparent",
                                    },
                                )
                            ],
                            color="#119DFF",
                            type="default",
                            fullscreen=False,
                        ),
                    ],
                ),
                html.Div(
                    dcc.Loading(
                        children=[
                            dcc.Graph(
                                id="loading-output",
                                figure=build_fig(),
                                className="mx-auto my-auto",
                                style={"width": 1200, "height": 500, "text-align": "center"},
                            )
                        ],
                        color="#119DFF",
                        type="default",
                        fullscreen=False,
                        className="mx-0 px-0 my-auto",
                    )
                ),
            ],
        )

        main_row = html.Div(
            # className = "row flex-fill d-flex justify-content-center",
            className="row justify-content-center",
            style={"height": "900"},
            id="main-row",
            children=[slider_col_1, slider_col_2, button_col, plot_col],
        )

        footer = html.Div(
            className="row navbar fixed-bottom navbar-dark bg-dark shadow-sm",
            id="footer",
            children=[],
        )

        # Build Layout
        app.layout = html.Div(
            style={"background-color": colors["background"]},
            className="container-fluid d-flex h-100 flex-column",
            children=[header, main_row, footer,],
        )

        # Slider Functionality
        for slider_key in slider_keys:
            slider_output_id = slider_key + "_output"
            slider_id = slider_key + "_slider"
            exec(
                "@app.callback(Output('"
                + slider_output_id
                + "', 'children'), [Input('"
                + slider_id
                + "', 'value')])\n"
                + "def "
                + "update_output_"
                + slider_key
                + "(value):"
                + "return "
                + "'"
                + slider_key
                + "'"
                + " + f' = {value}'"
            )

        # Button functionality
        @app.callback(Output("loading-output", "figure"), [Input("loading-button", "n_clicks")])
        def load_output(n_clicks):
            time.sleep(5)
            return build_fig()

        return app

    def run_app(self):
        """Run app and display result inline in the notebook"""
        return self.app.run_server(mode="inline", width="100%", height="880")

