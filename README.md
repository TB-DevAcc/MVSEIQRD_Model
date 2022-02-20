# MVSEIQRD Model

*Maternity-derived Immunity - Vaccinated - Susceptible - Exposed - Infectious - Quarantined - Recovered - Dead*

An extended SEIR model.

<p align="center">
    <img src="assets/full_plot.png" witdh="50%">
<p>

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
<img align="left" src="https://img.shields.io/github/last-commit/TB-DevAcc/MVSEIQRD_Model" alt="Last-commit Badge">

## Basic Overview

A simulation of the coronavirus pandemic in Germany based on the SEIR model. 
    
The main module of the software is the Model, which runs the simulation, initializes the controller and enables/updates the visualization.

The simulation itself takes place in the simulator, which constructs a system out of the single differential equations. It is able to build the ODE system modularly and then solves them as an initial value problem using the SciPy library. The modular design is noteworthy here, allowing the model to be augmented or reduced to a different epidemiological model. Although the implementation was done for the MVSEIQRD model, it can be easily reduced to the SEIR model, for example.
    
The controller contains the current data of the parameters and reads data from existing files together with the DataHandler. 

The view contains the dashboard and is able to display both the real and simulated data and their progressions as a graph. 
    
## :date: Timeframe for the project

The project took place between December 2021 and March 2022.

## :wrench: Usage

To start the app call the model class and let it start the Dashboard. Note that the Dashboard is meant to be run in a Jupyter Notebook.
```python
from app.model import Model
    
model = Model()
    
model.run_app() # Shows the Dashboard that enables tweaking parameters, running the simulation and plotting the results
```
You are then able to tweak hyperparameters to review different scenarios on the Dashboard.

Simply running the simulation and plotting the results is also possible:   
```python
model.run()
model.plot()
```
    
For a complete presentation and further explanations see the [Jupyter Notebook](MVSEIQRD_Modell_Jupyter.ipynb).

## :boy: Authors

**Tobias Becher**<br>
**Maximilian Fischer**<br>
**Artur Safenreiter**<br>

## :pray: Acknowledgments

This project was part of the course "Scientific Programming in Python" at University of Hagen. 
    
The data is taken from [Robert Koch Institut](https://github.com/robert-koch-institut).

## 📝 License

Copyright © 2021 [Tobias Becher](https://github.com/TB-DevAcc). <br/>
This project is [MIT](https://github.com/TB-DevAcc/SEIQRS/blob/master/LICENSE) licensed.
