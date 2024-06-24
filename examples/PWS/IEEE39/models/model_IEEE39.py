from pathlib import Path

import julia
from julia import Main

WORK_DIR = Path(__file__).absolute().parent

# Run IEEE39_SEModel.jl
j = julia.Julia()
m = j.include(str(WORK_DIR / "IEEE39_SEModel.jl"))

# Redefine the variables/functions for ease of use
h = Main.IEEE39_SEModel.h
x0 = Main.IEEE39_SEModel.x0
y = Main.IEEE39_SEModel.y

nparams = len(x0)
npreds = len(y)
