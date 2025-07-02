from naming_model import NamingModel
from mesa.visualization.modules import CanvasGrid
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import Slider
from mesa.visualization.modules import ChartModule

# Dimensions of the grid (number of cells x number of cells)
width = 30
height = 30
n_agents = [400, 100, 900, 5]  # default, min, max, increment


def agent_portrayal(agent):
    portrayal = {"Shape": "circle", "Filled": "true", "Color": "red",
                 "r": 0.5, "Layer": 0}
    # If there is an interaction occurring in that cell, paint a green circle, otherwise
    # a red circle
    if len(agent.model.grid.get_cell_list_contents([agent.pos])) > 1:
        portrayal["Color"] = "green"
    return portrayal


grid = CanvasGrid(agent_portrayal, width, height, 500, 500)
tot_graph = ChartModule(
    [{"Label": "Total_Words", "Color": "Red"}],
    data_collector_name='datacollector'
)
diff_graph = ChartModule(
    [{"Label": "Different_Words", "Color": "Blue"}],
    data_collector_name='datacollector'
)
prob_graph = ChartModule(
    [{"Label": "Prob_Success", "Color": "Green"}],
    data_collector_name='datacollector'
)
number_of_agents_slider = Slider(
    "Number of Agents", n_agents[0], n_agents[1], n_agents[2], n_agents[3])
server = ModularServer(NamingModel,
                       [grid, tot_graph, diff_graph, prob_graph],
                       "Minimal Naming Game",
                       {"n": number_of_agents_slider, "width": width, "height": height}
                       )
