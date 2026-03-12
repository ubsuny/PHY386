# Copilot Code Review Instructions — PHY386 Computational Physics

This is an undergraduate physics course. Students are learning Python, scientific computing, and professional software practices. Review their homework submissions with this context in mind.

## Tone
- Be encouraging and educational, not harsh
- Explain *why* something should change, not just *what* to change
- If code works but could be improved, frame it as a suggestion, not a requirement
- Praise good practices when you see them (clear variable names, good docstrings, proper units)

## Priority (review in this order)
1. **Physics correctness** — Are equations implemented correctly? Are units consistent? Do boundary conditions match the problem statement?
2. **Type annotations** — Every function must have type annotations on all parameters and the return type
3. **Docstrings** — Every function must have a NumPy-style docstring with Parameters and Returns sections, including units
4. **Plot quality** — Axes must be labeled with units, figures should have titles, colorbars where appropriate
5. **Code quality** — Readable variable names, no magic numbers without comments, reasonable structure

## What NOT to flag
- Do not flag minor style preferences (single vs double quotes, blank lines)
- Do not suggest advanced Python patterns students haven't learned (decorators, dataclasses, generators)
- Do not suggest external libraries beyond numpy, scipy, matplotlib, pandas
- Do not rewrite working code just to make it more "Pythonic"

## Debugging Challenge — DO NOT REVIEW
- Some homework notebooks contain a **Debugging Challenge** section with intentionally buggy code
- Code cells or functions marked with `# BUGGY CODE` or `# DEBUGGING CHALLENGE` contain **deliberate errors** that students must find themselves
- **Do NOT flag, fix, or hint at the bugs in these sections** — that defeats the purpose of the exercise
- Only review the student's **corrected** code (in cells below the buggy code) for type annotations and docstrings

## Physics-specific checks
- Verify finite difference stencils have correct signs
- Check that stability conditions are enforced for time-stepping schemes
- Verify boundary conditions match the problem (Dirichlet vs Neumann)
- Check that physical constants and units are reasonable (not off by orders of magnitude)
- Flag if a numerical result contradicts a known analytical limit
