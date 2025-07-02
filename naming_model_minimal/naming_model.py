from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
import numpy as np
from random import randint


# If this variable is set to True, the behaviour is as described in the article and the same
# graphs get created for total words, different words and probability of success
random_interactions = False
# This file is present in each Linux distribution, contains over 100000 words
word_file = "./words"
words = open(word_file).read().splitlines()
num_words = len(words)


class PersonAgent(Agent):
    """The agent class of the model"""
    def __init__(self, unique_id, model):
        # Call the superclass constructor
        super().__init__(unique_id, model)
        # The inventory of the agent, contains the word he has learned
        self.inventory = np.array([])
        # A member that will make it easy to have a listener and a hearer in each interaction
        self.will_listen = False

    def move(self):
        # The agent is almost forced to move, so we don't include the center
        possible_steps = self.model.grid.get_neighborhood(
            self.pos, moore=True, include_center=False)
        # We shuffle the array of possible positions to have at each call a random
        # direction of movement
        np.random.shuffle(possible_steps)
        # We cycle through all the possible steps, as soon as the agent finds a cell with less than two
        # occupants, they move there
        for pos in possible_steps:
            if len(self.model.grid.get_cell_list_contents([pos])) < 2:
                self.model.grid.move_agent(self, pos)
                return

    def speak(self, other_agent):
        # If it's the first time this agent interacts, they will have an empty inventory
        # and will need to generate a random word
        if np.size(self.inventory) == 0:
            new_word = words[randint(0, num_words-1)]
            # If the word is already present generate a new one
            while new_word in self.model.global_inventory:
                new_word = words[randint(0, num_words-1)]
            # Add the generated word to the global inventory and to the agent inventory
            self.model.global_inventory = np.append(self.model.global_inventory, new_word)
            self.inventory = np.append(self.inventory, new_word)
        # The number of total interactions goes up by one
        self.model.num_interactions += 1
        # The speaker selects a random word among their inventory
        word = self.random.choice(self.inventory)
        # If the word is not present in the hearer's inventory, it gets added
        # and the interaction ends here
        if word not in other_agent.inventory:
            other_agent.inventory = np.append(other_agent.inventory, word)
            # That word must also be added to the global inventory because now there are
            # two of them
            self.model.global_inventory = np.append(self.model.global_inventory, word)
            return
        # If we reach this part of the function, then we have a successful interaction
        self.model.successful_interactions += 1
        # All the words must be erased from the global and both the agent's inventories
        for elem in self.inventory:
            # Search for the index of the word in the global inventory
            indexes = np.where(self.model.global_inventory == elem)[0]
            if np.size(indexes) > 0:
                # Delete word at given index
                self.model.global_inventory = np.delete(self.model.global_inventory, indexes[0])
        # Do the same thing with the other agent's inventory
        for elem in other_agent.inventory:
            indexes = np.where(self.model.global_inventory == elem)[0]
            if np.size(indexes) > 0:
                self.model.global_inventory = np.delete(self.model.global_inventory, indexes[0])
        # Both the agent's inventories must be reduced to the single word they now share
        self.inventory = np.array([word])
        other_agent.inventory = np.array([word])
        # Now we add this word to the global inventory (doubled because two agents have it)
        self.model.global_inventory = np.append(self.model.global_inventory, [word, word])

    def step(self):
        # If we want to check the functioning of the model as described in the article, we use
        # random interaction mechanism (the topology becomes irrelevant)
        if random_interactions:
            # Select a random agent among the list of all agents
            other_agent = self.random.choice(self.model.schedule.agents)
            # Speak to that agent
            self.speak(other_agent)
            # Do nothing else
            return
        # First thing, the agent moves
        self.move()
        # Checks whether they have a cellmate or not
        cellmate = self.model.grid.get_cell_list_contents([self.pos])
        # Must delete themselves among the list of cellmates, this way the first position
        # in the array cellmate is the eventual cellmate
        cellmate.remove(self)
        # If there isn't any cellmate, do nothing
        if not np.any(cellmate):
            return
        # If they are a listener, they do nothing but restoring their will_listen member to False
        if self.will_listen:
            self.will_listen = False
            return
        # If they are a speaker, they speak to their cellmate, that becomes a listener
        self.speak(cellmate[0])
        cellmate[0].will_listen = True


class NamingModel(Model):
    """The model class"""
    def __init__(self, n, width, height):
        # Call the superclass constructor
        super().__init__()
        self.running = True
        self.num_agents = n
        # Members to calculate the probability of success in the interactions
        self.num_interactions = 0
        self.successful_interactions = 0
        # Standard grid and schedule instantiations
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        for i in range(self.num_agents):
            # Place agents on the grid, maximum two for each cell
            a = PersonAgent(i, self)
            self.schedule.add(a)
            while a.pos is None:
                x = self.random.randrange(self.grid.width)
                y = self.random.randrange(self.grid.height)
                # Try placing agents as long as there are no more than two in each cell
                if len(self.grid.get_cell_list_contents([(x, y)])) < 2:
                    self.grid.place_agent(a, (x, y))
        # The global inventory is to keep trace of all the words that are stored in the agent's vocabulary
        self.global_inventory = np.array([])
        self.datacollector = DataCollector(
            model_reporters={"Total_Words": calculate_total_words,
                             "Different_Words": calculate_different_words,
                             "Prob_Success": prob_inter},
            agent_reporters={}
        )

    def step(self):
        # Every time we call the step function, we must restore these variables to 0 so that we evaluate
        # the probability of success at each step
        self.successful_interactions = 0
        self.num_interactions = 0
        self.schedule.step()
        # After executing one step, collect the data
        self.datacollector.collect(self)


# Functions to calculate the variables we want to keep track of
def calculate_total_words(model):
    return len(model.global_inventory)


def calculate_different_words(model):
    return len(np.unique(model.global_inventory))


def prob_inter(model):
    if model.num_interactions == 0:
        return 0.0
    probability = model.successful_interactions/model.num_interactions
    return probability
