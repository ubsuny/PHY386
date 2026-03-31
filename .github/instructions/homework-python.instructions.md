---
applyTo: "2026/HW/**/*.{ipynb,py}"
---

# Homework Notebook Review — Python in Jupyter

## Type Annotations (required from HW4 onward)
- Every function must have type annotations: `def f(x: float) -> np.ndarray:`
- Check return type matches what the function actually returns
- Common types: `float`, `int`, `bool`, `str`, `np.ndarray`, `tuple[np.ndarray, int]`
- Flag `-> float` when the function returns an array, or vice versa

## Docstrings (required from HW2 onward)
- NumPy-style with Parameters and Returns sections
- Each parameter should include units in the description
- Example:
  ```python
  def solve(V0: float, R: float) -> np.ndarray:
      """Solve the circuit for branch currents.

      Parameters
      ----------
      V0 : float
          Source voltage (V).
      R : float
          Resistance (Ohm).

      Returns
      -------
      np.ndarray
          Branch currents (A).
      """
  ```

## Plotting
- Axes must have labels WITH units: `ax.set_xlabel('Position (m)')`
- Figures should use `figsize=(6, 4)` for single plots
- Colormaps: viridis for sequential data, RdBu for diverging (e.g., electric potential)
- Flag use of jet colormap — suggest viridis or cividis instead
- Tick marks should point inward: `ax.tick_params(direction='in', top=True, right=True)`

## Debugging Challenge — DO NOT REVIEW
- Code cells marked with `# BUGGY CODE` or `# DEBUGGING CHALLENGE` contain **deliberate errors**
- **Do NOT flag, fix, or hint at bugs in these sections** — students must find them independently
- Only review the student's corrected version (in separate cells below)

## Common Physics Checks
- Verify finite difference stencils and sign conventions
- Check stability conditions for time-stepping schemes
- Verify boundary conditions match the problem statement
- Check that physical constants and units are reasonable

## Commit Messages
- Flag vague commit messages if visible: "update", "fix", "stuff" are not acceptable
- Good examples: "Implement Jacobi solver for Laplace equation", "Add convergence plot"
