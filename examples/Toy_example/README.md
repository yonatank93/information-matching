# Toy model example -- Simple weather model

This toy example is mainly used to introduce the interface and the general workflow to run
the information-matching calculation. To illustrate the process, we use a simple periodic
function to model the data of monthly averaged atmospheric pressure difference between
Easter Island and Darwin, Australia.



## Content

* ENSO.txt: The average atmospheric pressure data that we will model.
* optimal_parameters.txt: The optimal parameters of the model from fitting the entire
  data.
* weather_oed.ipynb: A Jupyter notebook that show how we can use information-matching
  to find the optimal data to collect as an optimal experimental design (OED) problem.
* weather_al.ipynb: A Jupyter notebook that shows how we can use information-matching in
  an active learning (AL) loop to optimize both the dataset and parameters.
