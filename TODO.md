For this branch, I want to make so that user can input their own convex objective
function. Here is my thought about how to do it (main changes would be in
`information_matching/convex_optimization.py`):
* Remove `l1norm_obj` argument.
  Replace it with `obj_fn` (callable) and `obj_fn_kwargs` (dict) arguments
  - We need to check if the `obj_fn` is a cvxpy function.
* Change the `_objective_fn` method, probably like
  ```bash
  def _objective_fn(self):
      scaled_weights = cp.multiply(self.wm, self.scale_weights.reshape((-1, 1)))
	  return self.obj_fn(scaled_weights, **self.obj_fn_kwargs)
  ```
* Create a unit test.
  Probably we can just put the test in `tests/test_convexopt.py`.
  For example, if we use l2-norm objective function instead of l1-norm, we would get many
  nonzero weights.
