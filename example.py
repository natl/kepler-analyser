"""
Small example of analysis using ModelGrid
"""
from __future__ import division, unicode_literals, print_function
import kepler_analyser as k

# subclass ModelGrid here if you need to change how the initial conditions
# file is read
# class ModelGrid(k.ModelGrid):
#     def load_key_table(self):
#         pass


def do_analysis():
    grid = k.ModelGrid("example_data", "xrba", "MODELS.txt", "example_output")
    grid.analyse_all()
    print("done")


if __name__ == "__main__":
    do_analysis()
