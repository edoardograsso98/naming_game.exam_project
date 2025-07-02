import mesa
from naming_model import NamingModel
from naming_model import only_one_opinion
from naming_server import width, height
from mesa.batchrunner import FixedBatchRunner
from naming_model import minority_counter, mixed_counter, general_counter
import matplotlib.pyplot as plt
import time

# Values used in the benchmark article
a_val = [327, 0.003, 0.336, width, height, 5]  # n. agents, committed fraction, beta, width, height, group size

params = {"n": a_val[0], "fraction": a_val[1], "beta": a_val[2], "width": a_val[3], "height": a_val[4],
          "groups_size": a_val[5]}

results = mesa.batch_run(NamingModel, parameters=params, iterations=1, max_steps=0, number_processes=1,
                         data_collection_period=1, display_progress=True)

my_model = NamingModel(a_val[0], a_val[1], a_val[2], a_val[3], a_val[4], a_val[5])
min_steps = 100
max_steps = 500000



start_time = time.time()
"""
runner = FixedBatchRunner(NamingModel, fixed_parameters={"n": a_val[0],
                                                         "fraction": a_val[1],
                                                         "beta": a_val[2],
                                                         "width": a_val[3],
                                                         "height": a_val[4],
                                                         "groups_size": a_val[5]},
                          iterations=2, max_steps=3,
                          model_reporters={"Minority_Opinion": minority_counter,
                                           "General_Opinion": general_counter,
                                           "Mixed_Opinion": mixed_counter},
                          )

runner.run_all()
"""


# for i in range(max_steps):
#     my_model.step()
#     if i % 50000 == 0 and i != 0:
#         print(i, "steps done")
#     if i >= min_steps and only_one_opinion(my_model):
#         break

print("--- %s seconds ---" % (time.time() - start_time))

df = my_model.datacollector.get_model_vars_dataframe()
df.plot()
plt.show()
