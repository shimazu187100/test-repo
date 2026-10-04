from cat_game_solver import MeowdokuSolver, sample_grid_colors, CAT

def test_solver_cat_count():
    """猫の配置状態の検証"""
    solver = MeowdokuSolver(sample_grid_colors)
    solver.solve_step_by_step()
    # 猫の座標セット(curr_cats)が正常に保持されているか確認
    assert isinstance(solver.curr_cats, set)

def test_solver_no_row_col_conflict():
    """確定した猫が同じ行・同じ列に重複していないか検証"""
    solver = MeowdokuSolver(sample_grid_colors)
    solver.solve_step_by_step()
    rows = [r for r, c in solver.curr_cats]
    cols = [c for r, c in solver.curr_cats]
    assert len(rows) == len(set(rows)), "行に猫が重複しています"
    assert len(cols) == len(set(cols)), "列に猫が重複しています"

def test_solver_logical_moves_generated():
    """推論処理によって確定手（Xまたは猫）が導出されたか検証"""
    solver = MeowdokuSolver(sample_grid_colors)
    solver.solve_step_by_step()
    assert len(solver.curr_xs) > 0 or len(solver.curr_cats) > 0, "確定手が導出されませんでした"
