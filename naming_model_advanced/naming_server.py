from naming_model import NamingModel
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import Slider
from mesa.visualization.modules import ChartModule
from naming_model import minority_opinion, mixed_opinion

# Dimensions of the grid (number of cells x number of cells)
width = 50
height = 50
# Arrays for slider initializations
n_agents = [327, 100, 1000, 1]  # default, min, max, increment
committed_f = [0.003, 0.001, 1.0, 0.001]
g_size = [5, 2, 10, 1]
beta_values = [0.336, 0.0, 1.0, 0.01]


def agent_portrayal(agent):
    portrayal = {"Shape": "circle", "Filled": "true", "Color": "blue",
                 "r": 0.5, "Layer": 0}
    if agent.active:
        portrayal["Color"] = "orange"
    return portrayal


grid = CanvasGrid(agent_portrayal, width, height, 500, 500)
opinions_graph = ChartModule(
    [{"Label": "Minority_Opinion", "Color": "Orange"},
     {"Label": "General_Opinion", "Color": "Blue"},
     {"Label": "Mixed_Opinion", "Color": "Red"}],
    data_collector_name='datacollector'
)

number_of_agents_slider = Slider(
    "Number of Agents", n_agents[0], n_agents[1], n_agents[2], n_agents[3])
minority_fraction_slider = Slider(
    "Fraction of Committed Agents", committed_f[0], committed_f[1], committed_f[2], committed_f[3])
groups_size_slider = Slider(
    "Maximum Group Size", g_size[0], g_size[1], g_size[2], g_size[3])
beta_slider = Slider(
    "Beta", beta_values[0], beta_values[1], beta_values[2], beta_values[3])
server = ModularServer(NamingModel,
                       [grid, opinions_graph],
                       "Naming Game with Groups",
                       {"n": number_of_agents_slider, "fraction": minority_fraction_slider,
                        "groups_size": groups_size_slider, "beta": beta_slider,
                        "width": width, "height": height}
                       )
