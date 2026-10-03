from cat_game_v34 import MeowdokuSolver, sample_grid_colors, CAT

def test_solver_cat_count():
    """猫が正しく全8箇所に確定配置されるか検証"""
    solver = MeowdokuSolver(sample_grid_colors)
    solver.solve_step_by_step()
    
    cats = [(r, c) for r in range(8) for c in range(8) if solver.grid[r][c].state == CAT]
    assert len(cats) == 8, f"猫の配置数が {len(cats)} 個です（8個期待）"

def test_solver_no_row_col_conflict():
    """確定した猫が同じ行・同じ列に重複していないか検証"""
    solver = MeowdokuSolver(sample_grid_colors)
    solver.solve_step_by_step()
    
    cat_coords = [(r, c) for r in range(8) for c in range(8) if solver.grid[r][c].state == CAT]
    rows = [r for r, c in cat_coords]
    cols = [c for r, c in cat_coords]
    
    assert len(set(rows)) == 8, "行に猫が重複しています"
    assert len(set(cols)) == 8, "列に猫が重複しています"

def test_solver_step_logs_generated():
    """推論ステップのログが正しく記録されているか検証"""
    solver = MeowdokuSolver(sample_grid_colors)
    solver.solve_step_by_step()
    
    assert len(solver.steps_log) > 0, "推論ログが空です"
