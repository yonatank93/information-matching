from pathlib import Path

import julia
from julia import Main

WORK_DIR = Path(__file__).absolute().parent

# Run IEEE14_SEModel.jl
j = julia.Julia()
m = j.include(str(WORK_DIR / "IEEE14_SEModel.jl"))

# Redefine the variables/functions for ease of use
h = Main.IEEE14_SEModel.h
x0 = Main.IEEE14_SEModel.x0
y = Main.IEEE14_SEModel.y

nparams = len(x0)
npreds = len(y)
