#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meowdoku / Cat Game Logic Solver v34
- 背理法における中間連鎖（ドミノ倒し）の可視化
- 連続する同色・同破綻理由マスのグループ化出力
- 末尾のタイポエラー修正版
"""

import copy

EMPTY = 0   # 未確定
CAT = 1     # 猫 (O)
CROSS = 2   # X (猫を置けない)

GRID_SIZE = 8

COLOR_NAMES = {
    0: "黄色", 1: "紫色", 2: "薄緑色", 3: "青色",
    4: "赤色", 5: "薄橙色", 6: "エメラルド色", 7: "ピンク色"
}

class Cell:
    def __init__(self, row, col, color):
        self.row = row
        self.col = col
        self.color = color
        self.state = EMPTY

class MeowdokuSolver:
    def __init__(self, grid_colors):
        self.grid_size = GRID_SIZE
        self.grid = [[Cell(r, c, grid_colors[r][c]) for c in range(GRID_SIZE)] for r in range(GRID_SIZE)]
        self.num_colors = len(set(c for row in grid_colors for c in row))
        self.steps_log = []

    def is_valid_coord(self, r, c):
        return 0 <= r < self.grid_size and 0 <= c < self.grid_size

    def get_neighbors(self, r, c):
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if self.is_valid_coord(nr, nc):
                    neighbors.append((nr, nc))
        return neighbors

    def apply_cat_eliminations(self, grid, sub_logs=None):
        changed = False
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if grid[r][c].state == CAT:
                    for i in range(self.grid_size):
                        if i != c and grid[r][i].state == EMPTY:
                            grid[r][i].state = CROSS
                            changed = True
                        if i != r and grid[i][c].state == EMPTY:
                            grid[i][c].state = CROSS
                            changed = True
                    for nr, nc in self.get_neighbors(r, c):
                        if grid[nr][nc].state == EMPTY:
                            grid[nr][nc].state = CROSS
                            changed = True
                    color = grid[r][c].color
                    for dr in range(self.grid_size):
                        for dc in range(self.grid_size):
                            if (dr != r or dc != c) and grid[dr][dc].color == color and grid[dr][dc].state == EMPTY:
                                grid[dr][dc].state = CROSS
                                changed = True
        return changed

    def apply_forced_cats(self, grid, sub_logs=None):
        changed = False

        for r in range(self.grid_size):
            empties = [(r, c) for c in range(self.grid_size) if grid[r][c].state == EMPTY]
            cats = [c for c in range(self.grid_size) if grid[r][c].state == CAT]
            if len(cats) == 0 and len(empties) == 1:
                er, ec = empties[0]
                grid[er][ec].state = CAT
                changed = True
                if sub_logs is not None:
                    sub_logs.append(f"第{r+1}行の空きが残り1マスになり R{er+1:02d}C{ec+1:02d} に猫が確定")

        for c in range(self.grid_size):
            empties = [(r, c) for r in range(self.grid_size) if grid[r][c].state == EMPTY]
            cats = [r for r in range(self.grid_size) if grid[r][c].state == CAT]
            if len(cats) == 0 and len(empties) == 1:
                er, ec = empties[0]
                grid[er][ec].state = CAT
                changed = True
                if sub_logs is not None:
                    sub_logs.append(f"第{c+1}列の空きが残り1マスになり R{er+1:02d}C{ec+1:02d} に猫が確定")

        for color in range(self.num_colors):
            empties = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)
                       if grid[r][c].color == color and grid[r][c].state == EMPTY]
            cats = [(r, c) for r in range(self.grid_size) for c in range(self.grid_size)
                     if grid[r][c].color == color and grid[r][c].state == CAT]
            if len(cats) == 0 and len(empties) == 1:
                er, ec = empties[0]
                grid[er][ec].state = CAT
                changed = True
                if sub_logs is not None:
                    c_name = COLOR_NAMES.get(color, f"色{color}")
                    sub_logs.append(f"「{c_name}」領域の空きが残り1マスになり R{er+1:02d}C{ec+1:02d} に猫が確定")

        return changed

    def get_contradiction_reason(self, grid):
        for r in range(self.grid_size):
            cats = sum(1 for c in range(self.grid_size) if grid[r][c].state == CAT)
            empties = sum(1 for c in range(self.grid_size) if grid[r][c].state == EMPTY)
            if cats == 0 and empties == 0:
                return f"「第{r+1}行」に猫を置ける空きマスが全滅"

        for c in range(self.grid_size):
            cats = sum(1 for r in range(self.grid_size) if grid[r][c].state == CAT)
            empties = sum(1 for r in range(self.grid_size) if grid[r][c].state == EMPTY)
            if cats == 0 and empties == 0:
                return f"「第{c+1}列」に猫を置ける空きマスが全滅"

        for color in range(self.num_colors):
            cats = sum(1 for r in range(self.grid_size) for c in range(self.grid_size)
                       if grid[r][c].color == color and grid[r][c].state == CAT)
            empties = sum(1 for r in range(self.grid_size) for c in range(self.grid_size)
                         if grid[r][c].color == color and grid[r][c].state == EMPTY)
            if cats == 0 and empties == 0:
                c_name = COLOR_NAMES.get(color, f"色{color}")
                return f"「{c_name}」領域に猫を置ける空きマスが全滅"

        return None

    def trace_contradiction_chain(self, start_r, start_c):
        test_grid = copy.deepcopy(self.grid)
        sub_logs = []
        
        test_grid[start_r][start_c].state = CAT

        loop = True
        while loop:
            loop = False
            if self.apply_cat_eliminations(test_grid, sub_logs):
                loop = True
            if self.apply_forced_cats(test_grid, sub_logs):
                loop = True

            reason = self.get_contradiction_reason(test_grid)
            if reason:
                return True, sub_logs, reason

        return False, [], None

    def solve_step_by_step(self):
        while True:
            changed = False

            if self.apply_forced_cats(self.grid):
                changed = True
                self.apply_cat_eliminations(self.grid)
                continue

            group_found = False
            for r in range(self.grid_size):
                for c in range(self.grid_size):
                    if self.grid[r][c].state == EMPTY:
                        is_broken, chain, reason = self.trace_contradiction_chain(r, c)
                        if is_broken:
                            target_color = self.grid[r][c].color
                            group_cells = []
                            group_chains = []

                            for c_next in range(c, self.grid_size):
                                if self.grid[r][c_next].state == EMPTY and self.grid[r][c_next].color == target_color:
                                    b_ok, ch, re = self.trace_contradiction_chain(r, c_next)
                                    if b_ok and re == reason:
                                        group_cells.append((r, c_next))
                                        group_chains.append((r, c_next, ch))

                            cell_strs = [f"R{gr+1:02d}C{gc+1:02d}" for gr, gc in group_cells]
                            color_name = COLOR_NAMES.get(target_color, "")
                            
                            for gr, gc in group_cells:
                                self.grid[gr][gc].state = CROSS

                            if len(group_cells) > 1:
                                range_str = f"{cell_strs[0]}〜{cell_strs[-1]}"
                                log_msg = f"【一括確定】X  -> {range_str} ({color_name}・計{len(group_cells)}マス)\n"
                            else:
                                log_msg = f"【確定】X  -> {cell_strs[0]} ({color_name})\n"

                            log_msg += "  └ [ドミノ倒しの検証プロセス]:\n"
                            ex_r, ex_c, ex_chain = group_chains[0]
                            log_msg += f"      1. 仮に R{ex_r+1:02d}C{ex_c+1:02d}({color_name}) 等に猫を置くと仮定\n"
                            idx = 2
                            for item in ex_chain:
                                log_msg += f"      {idx}. {item}\n"
                                idx += 1
                            log_msg += f"      {idx}. 結果: {reason}（破綻）\n"
                            log_msg += f"  └ [結論] 上記の連鎖により詰むため、対象マス（{', '.join(cell_strs)}）はすべて X 確定"

                            self.steps_log.append(log_msg)
                            changed = True
                            group_found = True
                            break
                if group_found:
                    break

            if not changed:
                break

    def print_result(self):
        print("【論理推論による解法ステップと連鎖詳細】\n")
        for i, log in enumerate(self.steps_log, 1):
            print(f"Step {i:02d}: {log}")
            print("-" * 70)

        cats = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                if self.grid[r][c].state == CAT:
                    c_name = COLOR_NAMES.get(self.grid[r][c].color, "")
                    cats.append((r+1, c+1, c_name))

        print("\n【最終結果】確定した猫の配置一覧（全8箇所）")
        print("=" * 45)
        for idx, (r, c, c_name) in enumerate(cats, 1):
            print(f"  猫 {idx}: R{r:02d}C{c:02d}  ({c_name})")
        print("=" * 45)

        print("\n【最終盤面】 (O: 猫, X: 置き不可)")
        for r in range(self.grid_size):
            row_str = ""
            for c in range(self.grid_size):
                if self.grid[r][c].state == CAT:
                    row_str += " O "
                elif self.grid[r][c].state == CROSS:
                    row_str += " X "
                else:
                    row_str += " . "
            print(f"R{r+1:02d} |{row_str}|")

sample_grid_colors = [
    [0, 0, 0, 0, 1, 2, 2, 2],
    [3, 0, 3, 0, 1, 1, 1, 2],
    [3, 3, 3, 0, 4, 4, 1, 5],
    [3, 6, 3, 0, 4, 4, 4, 5],
    [3, 6, 3, 3, 4, 4, 5, 5],
    [3, 6, 6, 3, 5, 5, 5, 5],
    [3, 7, 6, 3, 3, 6, 6, 6],
    [7, 7, 6, 6, 6, 6, 6, 6]
]

if __name__ == "__main__":
    solver = MeowdokuSolver(sample_grid_colors)
    solver.solve_step_by_step()
    solver.print_result()
