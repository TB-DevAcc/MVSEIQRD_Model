# App logic
class View:

    def seir_plot(seir, t, names):
        """

        Parameters
        ----------
        t
        names : list of Strings

        Returns : figure
        -------

        """
        fig = go.Figure()
        for i in range(len(seir)):
            fig.add_trace(go.Line(x=t, y=np.sum(seir[i], axis=1),
                                  mode='lines',
                                  name=names[i]))
            fig.update_layout(
                autosize=False,
                width=700,
                height=700)
        return fig

    def seiqrds_plot(seiqrd_s, t, names):
        """

        Parameters
        ----------
        t
        names : list of Strings

        Returns : figure
        -------

        """
        fig = go.Figure()
        for i in range(len(seiqrd_s)):
            fig.add_trace(go.Line(x=t, y=np.sum(seiqrd_s[i], axis=(1, 2)),
                                  mode='lines',
                                  name=names[i]))
            fig.update_layout(
                autosize=False,
                width=700,
                height=700)
        return fig

    def covid_plot(df):

        """
         Parameters
        ----------
        df : DataFrame
        Returns fig
        -------

        """

        fig = go.Figure()
        fig.add_trace(go.Line(x=df["Meldedatum"], y=df["AnzahlFall"],
                              mode='lines',
                              name='Anzahl Fälle'))
        fig.add_trace(go.Line(x=df["Meldedatum"], y=df["seven-day-average"],
                              mode='lines',
                              name='7-Tage Mittelwert'))
        return fig


