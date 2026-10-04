#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Meowdoku (Cat Puzzle) Solver V29
"""

import sys
import os
import cv2
import argparse
import glob
import json
import numpy as np
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEBUG_DIR = os.path.join(SCRIPT_DIR, 'temp_py')
os.makedirs(DEBUG_DIR, exist_ok=True)

def board_region(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] > 25).astype(np.uint8)) * 255
    h, w = mask.shape
    k = max(3, min(15, int(round(min(h, w) * .01))))
    k += (k % 2 == 0)
    d = cv2.dilate(mask, np.ones((k, k), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(d, 8)
    cand = [stats[i] for i in range(1, n) if stats[i][2] >= w * .35 and stats[i][3] >= h * .15]
    if not cand: return (int(w * 0.03), int(h * 0.30), int(w * 0.94), int(w * 0.94)), mask
    rx, ry, rw, rh = max(cand, key=lambda z: z[2] * z[3])[:4]
    side = max(rw, rh)
    return (rx, ry, side, side), mask

def runs(v, t):
    ids = np.where(v > t)[0]
    if not len(ids): return []
    out, s, p = [], int(ids[0]), int(ids[0])
    for z in ids[1:]:
        z = int(z)
        if z == p + 1: p = z
        else:
            out.append((s, p))
            s = p = z
    out.append((s, p))
    return out

def estimate_grid_size(mask, rect):
    x, y, w, h = rect
    v_sum = mask[y:y+h, x:x+w].sum(axis=0)
    n = len(runs(v_sum, np.max(v_sum) * 0.15))
    return n if 7 <= n <= 11 else 9

def detect_grid(mask, rect, grid_size):
    _, _, w, h = rect
    cw, ch = w / grid_size, h / grid_size
    xr = [(int(i * cw), int((i + 1) * cw - 1)) for i in range(grid_size)]
    yr = [(int(i * ch), int((i + 1) * ch - 1)) for i in range(grid_size)]
    return xr, yr

def cell_roi(img, rect, xr, yr):
    x, y, _, _ = rect
    xa, xb = xr; ya, yb = yr
    mx, my = max(1, int((xb-xa) * .08)), max(1, int((yb-ya) * .08))
    return img[y+ya+my:y+yb+1-my, x+xa+mx:x+xb+1-mx]

def get_corner_bg_color(roi):
    ch, cw, _ = roi.shape
    pts = np.vstack([
        roi[:int(ch*.3), :int(cw*.3)].reshape(-1, 3),
        roi[:int(ch*.3), -int(cw*.3):].reshape(-1, 3),
        roi[-int(ch*.3):, :int(cw*.3)].reshape(-1, 3),
        roi[-int(ch*.3):, -int(cw*.3):].reshape(-1, 3)
    ])
    return np.median(pts, axis=0)

def symbol(roi, bg_bgr):
    diff = np.linalg.norm(roi.astype(np.float32) - bg_bgr, axis=2)
    mark_mask = diff > 40.0
    if mark_mask.mean() < 0.012: return ''
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[mark_mask]
    s, v = hsv[:, 1], hsv[:, 2]
    white = (s < 60) & (v > 180)
    black = v < 100
    if float(black.mean()) > 0.10 and float(white.mean()) > 0.04: return 'C'
    if float((((hsv[:, 0] < 10) | (hsv[:, 0] > 170)) & (s > 90) & (v > 90)).mean()) > 0.15: return 'RX'
    if float(white.mean()) > 0.20: return 'WX'
    return ''

def recognize(img):
    rect, mask = board_region(img)
    gsize = estimate_grid_size(mask, rect)
    xr, yr = detect_grid(mask, rect, gsize)
    
    rx, ry, rw, rh = rect
    cropped = img[ry:ry+rh, rx:rx+rw]
    cv2.imwrite(os.path.join(DEBUG_DIR, 'board_cropped.png'), cropped)
    
    grid_img, rec_img = img.copy(), img.copy()
    cells, cats, red_xs, white_xs, samples = [], [], [], [], []
    for r, y_bound in enumerate(yr):
        for c, x_bound in enumerate(xr):
            samples.append(get_corner_bg_color(cell_roi(img, rect, x_bound, y_bound)))
            
    samples_lab = cv2.cvtColor(np.array(samples, dtype=np.uint8).reshape(-1, 1, 3), cv2.COLOR_BGR2Lab).astype(np.float32)
    _, labels, _ = cv2.kmeans(samples_lab, gsize, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1), 50, cv2.KMEANS_PP_CENTERS)
    
    for i, (r, y_bound) in enumerate(zip(range(gsize), yr)):
        for j, (c, x_bound) in enumerate(zip(range(gsize), xr)):
            idx = i * gsize + j
            cid = int(labels[idx].ravel()[0])
            sym = symbol(cell_roi(img, rect, x_bound, y_bound), samples[idx])
            
            cells.append({'row': r, 'col': c, 'color_id': cid, 'symbol': sym})
            if sym == 'C': cats.append((r, c))
            elif sym in ('RX', 'WX'): white_xs.append((r, c))
            
            xa, xb, ya, yb = x_bound[0], x_bound[1], y_bound[0], y_bound[1]
            cv2.rectangle(grid_img, (rx+xa, ry+ya), (rx+xb, ry+yb), (0, 255, 0), 1)
            cx, cy = rx + (xa + xb) // 2, ry + (ya + yb) // 2
            lbl = f"{cid}:{sym}" if sym else f"{cid}"
            cv2.putText(rec_img, lbl, (cx - 15, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    cv2.imwrite(os.path.join(DEBUG_DIR, 'debug_grid.png'), grid_img)
    cv2.imwrite(os.path.join(DEBUG_DIR, 'debug_recognized.png'), rec_img)
    
    with open(os.path.join(DEBUG_DIR, 'cells_data.json'), 'w', encoding='utf-8') as f:
        json.dump({'grid_size': gsize, 'cells': cells}, f, ensure_ascii=False, indent=2)
            
    return {'grid': {'rows': gsize, 'cols': gsize}, 'cells': cells, 'raw_cats': cats, 'raw_xs': white_xs}

def solve(rec, nr, nc):
    colors = defaultdict(list)
    fixedcats = []
    fixedx = set(rec['raw_xs'])
    fixed_by_color = {}
    
    for item in rec['cells']:
        r, c, cid, sym = item['row'], item['col'], item['color_id'], item['symbol']
        colors[cid].append((r, c))
        if sym == 'C':
            fixedcats.append((r, c))
            fixed_by_color[cid] = (r, c)
            
    dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    cand = {cid: [p for p in lst if p not in fixedx and (cid not in fixed_by_color or p == fixed_by_color[cid])] for cid, lst in colors.items()}
    usedr = {r for r, c in fixedcats}
    usedc = {c for r, c in fixedcats}
    placed = list(fixedcats)
    sols = []

    def dfs(rem_colors):
        if len(sols) >= 50: return
        if not rem_colors: 
            sols.append(tuple(sorted(placed)))
            return
            
        best_cid, best_arr = min(((cid, [p for p in cand[cid] if p[0] not in usedr and p[1] not in usedc and not any((p[0]+dr, p[1]+dc) in placed for dr, dc in dirs)]) for cid in rem_colors), key=lambda x: len(x[1]))
        if not best_arr: return
        
        for r, c in best_arr:
            usedr.add(r); usedc.add(c); placed.append((r, c))
            dfs([x for x in rem_colors if x != best_cid])
            placed.pop(); usedr.remove(r); usedc.remove(c)
            
    dfs([c for c in colors if c not in fixed_by_color])
    return sols

def find_logical_moves(cells_data, known_cats, known_xs, nr, nc):
    color_cells, cid_map = defaultdict(list), {}
    for item in cells_data['cells']:
        r, c, cid = item['row'], item['col'], item['color_id']
        color_cells[cid].append((r, c))
        cid_map[(r, c)] = cid

    known_cats = set(known_cats)
    known_xs = set(known_xs)
    dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    for r, c in list(known_cats):
        for i in range(nr): known_xs.add((i, c))
        for j in range(nc): known_xs.add((r, j))
        for dr, dc in dirs: known_xs.add((r+dr, c+dc))
        for cr, cc in color_cells[cid_map[(r, c)]]: known_xs.add((cr, cc))
    known_xs -= known_cats
    move_groups = []

    for cid, clist in color_cells.items():
        avail = [p for p in clist if p not in known_xs and p not in known_cats]
        if len(avail) == 1 and not any(p in known_cats for p in clist):
            return [[(avail[0], 'C', f'色(ID:{cid})内で残された唯一の空きマスのため')]]

    for cid, clist in color_cells.items():
        avail = [p for p in clist if p not in known_xs and p not in known_cats]
        if not avail: continue
        rows, cols = set(r for r,c in avail), set(c for r,c in avail)
        if len(cols) == 1 and len(avail) > 1:
            group = [( (r, list(cols)[0]), 'X', f'色(ID:{cid})がC{list(cols)[0]+1:02d}列に限定配置されているため' ) for r in range(nr) if (r, list(cols)[0]) not in clist and (r, list(cols)[0]) not in known_xs and (r, list(cols)[0]) not in known_cats]
            if group: move_groups.append(group)
        if len(rows) == 1 and len(avail) > 1:
            group = [( (list(rows)[0], c), 'X', f'色(ID:{cid})がR{list(rows)[0]+1:02d}行に限定配置されているため' ) for c in range(nc) if (list(rows)[0], c) not in clist and (list(rows)[0], c) not in known_xs and (list(rows)[0], c) not in known_cats]
            if group: move_groups.append(group)
    if move_groups: return move_groups

    for r in range(nr):
        avail = [c for c in range(nc) if (r, c) not in known_xs and (r, c) not in known_cats]
        if not avail: continue
        cids = set(cid_map[(r, c)] for c in avail)
        if len(cids) == 1:
            cid = list(cids)[0]
            group = [(p, 'X', f'R{r+1:02d}行の空きが全て色(ID:{cid})内にあるため、ブロックの他マスをXに確定') for p in color_cells[cid] if p[0] != r and p not in known_xs and p not in known_cats]
            if group: return [group]

    for c in range(nc):
        avail = [r for r in range(nr) if (r, c) not in known_xs and (r, c) not in known_cats]
        if not avail: continue
        cids = set(cid_map[(r, c)] for r in avail)
        if len(cids) == 1:
            cid = list(cids)[0]
            group = [(p, 'X', f'C{c+1:02d}列の空きが全て色(ID:{cid})内にあるため、ブロックの他マスをXに確定') for p in color_cells[cid] if p[1] != c and p not in known_xs and p not in known_cats]
            if group: return [group]

    def is_broken(t_cats, t_xs):
        for r in range(nr):
            if not any(cr == r for cr, cc in t_cats) and sum(1 for c in range(nc) if (r, c) not in t_xs) == 0: return True
        for c in range(nc):
            if not any(cc == c for cr, cc in t_cats) and sum(1 for r in range(nr) if (r, c) not in t_xs) == 0: return True
        for cid, clist in color_cells.items():
            if not any(p in t_cats for p in clist) and sum(1 for p in clist if p not in t_xs) == 0: return True
        return False

    for r in range(nr):
        for c in range(nc):
            p = (r, c)
            if p in known_xs or p in known_cats: continue
            
            test_cats = set(known_cats) | {p}
            test_xs = set(known_xs)
            for i in range(nr): test_xs.add((i, c))
            for j in range(nc): test_xs.add((r, j))
            for dr, dc in dirs: test_xs.add((r+dr, c+dc))
            for cr, cc in color_cells[cid_map[p]]: test_xs.add((cr, cc))
            
            if is_broken(test_cats, test_xs):
                return [[(p, 'X', f'ここにネコを置くと他の行・列・色の配置場所が消滅し矛盾するため（背理法）')]]
                
            if is_broken(known_cats, set(known_xs) | {p}):
                return [[(p, 'C', f'ここをXにすると他の行・列・色の配置場所が消滅し矛盾するため（背理法）')]]

    return []

def format_cells(cells):
    parts, processed = [], set()
    for r in sorted(set(r for r, c in cells)):
        cols = sorted([c for cr, c in cells if cr == r])
        start = prev = cols[0]
        for n in cols[1:] + [999]:
            if n == prev + 1: prev = n
            else:
                if prev > start:
                    parts.append(f"R{r+1:02d}C{start+1:02d}-R{r+1:02d}C{prev+1:02d}")
                    for c in range(start, prev+1): processed.add((r, c))
                start = prev = n
    for r, c in sorted(cells):
        if (r, c) not in processed: parts.append(f"R{r+1:02d}C{c+1:02d}")
    return ", ".join(parts)

def main():
    parser = argparse.ArgumentParser(description="Meowdoku Solver")
    parser.add_argument("-onlynext", action="store_true", help="全解探索をスキップし、最初の1手のみを出力する")
    parser.add_argument("image_path", nargs="?")
    args = parser.parse_args()

    path = args.image_path
    if not path:
        search_dirs = [
            '/storage/emulated/0/Download',
            '/storage/emulated/0/Pictures/Screenshots',
            '/storage/emulated/0/DCIM/Screenshots',
            '.'
        ]
        exts = ['*.png', '*.jpg', '*.jpeg']
        files = []
        for d in search_dirs:
            if os.path.exists(d):
                for ext in exts:
                    files.extend(glob.glob(os.path.join(d, ext)))
        path = max(files, key=os.path.getmtime) if files else None
        if not path:
            print("画像が見つかりません。")
            return 1
        print(f'-> 自動選択: {os.path.basename(path)}')

    img = cv2.imread(path)
    if img is None:
        print(f"画像の読み込みに失敗しました: {path}")
        return 1

    rec = recognize(img)
    nr, nc = rec['grid']['rows'], rec['grid']['cols']
    
    print('\n【デバッグ出力】')
    print(f' - 盤面サイズ: {nr}x{nc}')

    print('\n【論理推論による連鎖手番の導出】')
    curr_cats = set(rec['raw_cats'])
    curr_xs = set(rec['raw_xs'])
    step = 1

    while True:
        logical_groups = find_logical_moves(rec, curr_cats, curr_xs, nr, nc)
        if not logical_groups:
            if step == 1:
                print('★ 高度な推論でも最初の手が確定しませんでした。')
            else:
                print(f'\n★ これ以上の確定手を論理推論のみで導出できませんでした（計 {step-1} ステップ完了）。')
            break

        group = logical_groups[0]
        cells = [(r, c) for (r, c), _, _ in group]
        target_sym = group[0][1]
        reason = group[0][2]

        print(f'ステップ {step}: 【確定】{target_sym} になるマス ({len(cells)}マス)')
        print(f'  - 対象: {format_cells(cells)}')
        print(f'  [理由] {reason}')

        for (r, c), sym_val, _ in group:
            if sym_val == 'C':
                curr_cats.add((r, c))
            elif sym_val == 'X':
                curr_xs.add((r, c))

        if args.onlynext:
            break

        step += 1

    # -onlynext 未指定時のみ、最後に全解探索を実行して結果を出力
    if not args.onlynext:
        sols = solve(rec, nr, nc)
        print('\n【全解探索 (すべての猫の位置の検討)】')
        if sols:
            print('★ 解の一例 (すべての猫の配置パターン):')
            ans_str = [f"R{r+1:02d}C{c+1:02d}" for r, c in sols[0]]
            for i in range(0, len(ans_str), 5):
                print("   " + ", ".join(ans_str[i:i+5]))
        else:
            print('★ 現在の盤面から到達できる解が見つかりません（矛盾が生じています）。')

if __name__ == '__main__':
    sys.exit(main())
