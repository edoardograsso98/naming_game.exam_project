from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
import numpy as np
import random
from random import uniform
from random import randint


# The two conflicting opinions agents can have
general_opinion = "A"
minority_opinion = "B"
mixed_opinion = [general_opinion, minority_opinion]


class PersonAgent(Agent):
    """The agent class of the model"""
    def __init__(self, unique_id, model, committed):
        # Call the superclass constructor
        super().__init__(unique_id, model)
        # This parameter indicates whether the agent is willing the change their mind
        self.is_committed = committed
        self.active = False
        self.mates = []
        # When initializing an agent, only committed agents have the minority opinion
        self.opinion = minority_opinion if self.is_committed else general_opinion

    def move(self):
        # The agent is almost forced to move, so we don't include the center
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False)
        # We shuffle the array of possible positions to have at each call a random direction of movement
        np.random.shuffle(possible_steps)
        # We cycle through all the possible steps, the number of maximum agents that can be in a single cell is a
        # parameter of the model
        for pos in possible_steps:
            # Only if the agent can't find a valid cell to move in they stay put
            if len(self.model.grid.get_cell_list_contents([pos])) < 1:
                self.model.grid.move_agent(self, pos)
                return

    def speak(self, other_agents):
        # If the speaker has a mixed opinion, they randomly chose one of the two words
        word = self.opinion if not self.opinion == mixed_opinion else mixed_opinion[randint(0, 1)]
        # We initialize an disagreement parameter to False
        disagreement = False
        for agent in other_agents:
            # If another agent has the same opinion as ours, we do nothing
            if word in agent.opinion:
                continue
            # Else we set the disagreement parameter to True and, if the agent disagreeing is not committed, we set
            # their opinion to mixed
            disagreement = True
            if not agent.is_committed:
                agent.opinion = mixed_opinion
        # If someone disagreed we do nothing
        if disagreement:
            return
        # Else, with probability of beta, we change everyone's opinion to the chosen one
        if uniform(0, 1) < self.model.propensity:
            self.opinion = word
            for agent in other_agents:
                agent.opinion = word

    def step(self):
        self.move()
        # If they are a speaker: first thing they move
        # Then gather the cellmates
        cellmates = []
        create_group(self, cellmates)
        if self in cellmates:
            cellmates.remove(self)
        self.mates = cellmates
        # Must delete themselves among the list of cellmates, this way the cellmates array contains the listeners
        # If there isn't any cellmate, do nothing
        if not np.any(cellmates):
            return
        # Speak to their cellmates, that become listeners
        self.speak(cellmates)


class NamingModel(Model):
    """The model class"""
    def __init__(self, n, fraction, beta, width, height, groups_size):
        # Call the superclass constructor
        super().__init__()
        self.running = True
        self.num_agents = n
        self.committed_fraction = fraction
        self.max_groups = groups_size
        self.propensity = beta
        # Standard grid and schedule instantiations
        self.grid = SingleGrid(width, height, True)
        self.schedule = RandomActivation(self)
        # Number of committed agents
        num_committed = round(n*fraction)
        # Number of non-committed agents
        num_general = n-num_committed
        # Calls the method for creating agents, we create the non-committed agents first, then the committed ones
        self.create_agents(num_general, 0, False)
        self.create_agents(num_committed, num_general, True)
        self.datacollector = DataCollector(
            model_reporters={"Minority_Opinion": minority_counter,
                             "General_Opinion": general_counter,
                             "Mixed_Opinion": mixed_counter},
            agent_reporters={}
        )

    def create_agents(self, num, already_existent, are_committed):
        for i in range(num):
            a = PersonAgent(already_existent+i, self, are_committed)
            self.schedule.add(a)
            self.grid.position_agent(a, x="random", y="random")

    def step(self):
        for agent in self.schedule.agents:
            agent.active = False
        actor = random.choice(self.schedule.agents)
        actor.step()
        actor.active = True
        for agent in actor.mates:
            agent.active = True
        self.datacollector.collect(self)


# Functions to calculate the variables we want to keep track of
def mixed_counter(model):
    return sum([agent.opinion == mixed_opinion for agent in model.schedule.agents])/model.num_agents


def minority_counter(model):
    return sum([agent.opinion == minority_opinion for agent in model.schedule.agents])/model.num_agents


def general_counter(model):
    return sum([agent.opinion == general_opinion for agent in model.schedule.agents])/model.num_agents


# Utility function to check whether there are still more than one opinion among the agents
def only_one_opinion(model):
    if mixed_counter(model) == 0:
        if general_counter(model) == 0:
            return True
        if minority_counter(model) == 0:
            return True
    return False


def create_group(speaker, global_mates):
    mates = speaker.model.grid.get_neighbors(speaker.pos, moore=True, include_center=False)
    to_exit = True
    for mate in mates:
        if not(mate in global_mates):
            to_exit = False
            global_mates.append(mate)
            if len(global_mates) >= speaker.model.max_groups-1:
                to_exit = True
                break
    if to_exit:
        return
    for agent in mates:
        create_group(agent, global_mates)
