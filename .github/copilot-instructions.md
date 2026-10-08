# Copilot instructions for this repository

## Project overview

This repository contains a Python-based Meowdoku / Cat Game solver. The project is organized around a workflow of:

- image-based board recognition
- color/shape extraction from cells
- logical deduction to place cats and X marks
- optional full-solution search for all valid layouts

The implementation and tests live under `40_SC/`, while the design documents live under `30_SDD/` and the repo-level context lives in `README.md`.

## Repository structure

- `README.md`: root-level project summary
- `30_SDD/`: design and specification documents describing the solver strategy
- `40_SC/`: Python implementation and test suite
  - `cat_game_solver.py`: solver, board recognition, logical deduction, CLI entrypoint
  - `test_cat_game.py`: regression and logic tests
- `50_DW/`: generated artifacts and logs
- `.github/workflows/test.yml`: CI definition for automated tests

## Build, test, and validation commands

There is no dedicated build system or lint config in this repository. The project uses `pytest` for validation.

Run the full test suite:

```bash
cd 40_SC
pytest
```

Run a single test by name:

```bash
cd 40_SC
pytest test_cat_game.py -k no_row_col_conflict
```

Run a single test file:

```bash
cd 40_SC
pytest test_cat_game.py
```

The GitHub Actions workflow runs the same project-local command:

```bash
cd 40_SC
pytest
```

## High-level architecture

The codebase is intentionally split by concern:

1. `30_SDD/` documents the intended solving strategy, especially the logical reasoning engine and contradiction propagation.
2. `40_SC/cat_game_solver.py` combines image processing and solver logic in one file:
   - `recognize(...)` extracts a board and cell colors/symbols from an input image
   - `MeowdokuSolver` performs step-by-step logical deduction and contradiction checks
   - `solve_all_solutions()` explores valid layouts when needed
   - `main()` provides the CLI entrypoint and prints recognized board state and solution candidates
3. `40_SC/test_cat_game.py` verifies solver invariants, including:
   - current cats remain a valid set
   - no row/column duplicate cat placement occurs
   - logical moves are generated during solving

## Key conventions specific to this repo

- The solver uses `color_id` values as the basis for region/group reasoning; many rules are expressed in terms of row, column, and color constraints instead of a generic constraint solver.
- The puzzle logic is written as deterministic inference plus contradiction checks, not as a generic SAT/ILP formulation.
- The project expects image input to be processed by OpenCV; helper functions in `cat_game_solver.py` build a debug output directory (`temp_py`) when running recognition.
- Logically derived moves are represented as groups of `(cell, symbol, reason)` tuples; this pattern is used by `find_logical_moves()` and `solve_step_by_step()`.
- The repo treats `30_SDD/*.md` as the source of truth for solver behavior and algorithm intent, while `40_SC/cat_game_solver.py` is the executable implementation.

## Working conventions for editing

- Prefer changing the solving logic in `40_SC/cat_game_solver.py` only when the behavior is governed by the design docs.
- When validating a logic change, use the existing pytest cases and keep the solver’s invariant checks in mind.
- If a fix affects board recognition or rule propagation, check the corresponding design description in `30_SDD/` before broad edits.
