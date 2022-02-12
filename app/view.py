import base64
from pathlib import Path

import dash_bootstrap_components as dbc
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydot
from dash import dcc, html
from dash.dependencies import Input, Output, State
from jupyter_dash import JupyterDash
from pyvis.network import Network


class View:
    def __init__(self, model):
        self.model = model
        self.network_svg_path = str(self._create_network_svg(return_b64=False))
        self.network_svg_b64 = (
            "data:image/svg+xml;base64," + str(self._create_network_svg(return_b64=True))[2:-1]
        )
        self.network_iframe_path = str(self._create_network_iframe())
        self.app = self._build_app()

    def plot(self, params=None, layout_dict: dict = None, show: bool = False):
        """
        Plots the course of the parameter development over time.

        Returns a plotly express figure
        """
        if not params:
            params = self.model.get_params()

        classes = self.model.translate_simulation_type()
        params = {k: np.sum(v, axis=(1, 2)) for k, v in params.items() if k in classes}
        df = pd.DataFrame(params)

        layout = {
            "title": "Simulation",
            "xaxis_title": r"$\text{Time } t \text{ in days}$",
            "yaxis_title": r"$\text{Number of people } n$",
            "legend_title_text": "Classes",
        }
        if layout_dict:
            for k, v in layout_dict:
                layout[k] = v

        fig = px.line(df)
        fig.update_layout(go.Layout(layout))

        if show:
            fig.show()
        return fig

    def _create_network_iframe(
        self,
        network_iframe_path=Path("assets/network.html"),
        dot_path=Path("data/param_graph.dot"),
    ):
        G = nx.DiGraph(nx.drawing.nx_pydot.read_dot(dot_path))
        net = Network(directed=True, notebook=True)
        net.from_nx(G)
        options = [
            """
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
        """,
            """
        var options = \
        {
        "nodes":{
            "font":{
                "background":"rgba(255,125,104,0.77)"
            }
        },
        "edges":{
            "color":{
                "inherit":true
            },
            "scaling":{
                "max":100
            },
            "font":{
                "size":9,
                "background":"rgba(255,255,255,0.90)"
            },
            "smooth":{
                "forceDirection":"none"
            }
        },
        "physics":{
            "minVelocity":0.75,
            "solver":"repulsion"
        }
        }
        """,
        ]
        net.set_options(options[1])
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
                                figure=px.line(width=800, height=500),
                                className="mx-auto my-auto",
                                style={"width": 800, "height": 500, "text-align": "center"},
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
            return self.plot(layout_dict={"width": 800, "height": 500})

        return app

    def run_app(self):
        """Run app and display result inline in the notebook"""
        return self.app.run_server(mode="inline", width="1400", height="880")

