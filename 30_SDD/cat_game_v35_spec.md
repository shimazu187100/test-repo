# Meowdoku Solver v35 仕様書（人間型推論の拡張）

## 1. 概要
v35 は「人間が解くときと同じように、なぜそこが猫なのか／なぜそこが X なのかを説明できる」ことを目的に、
世界の類似パズル（LinkedIn Queens 系ソルバー、数独の解法理論）のテクニックを調査し、
v34 までに不足していた推論アルゴリズムを追加したバージョンである。

## 2. 調査した手法と既存実装との対応

| 手法 | 出典の考え方 | v34まで | v35 |
|---|---|---|---|
| 基礎波及（猫の同行・同列・周囲・同色をX） | Queens/数独共通 | 実装済 | ― |
| 唯一マス（色・行・列の Hidden Single） | Queens Solver "Region/Row/Column singleton"、数独 Hidden Single | 実装済 | ― |
| 限定配置（Pointing） | Queens Solver "Region forced row/col"、数独 Pointing Pair | 実装済 | ― |
| 交差推論（Claiming） | 数独 Box-Line Reduction の色版 | 実装済 | ― |
| 浅い背理法（仮定直後の即時排除のみで判定） | ― | 実装済（限界あり） | 連鎖伝播付きへ置換 |
| **連鎖伝播付き背理法（Lookahead）** | Queens Solver "Lookahead / Elimination / Double-Block"、数独 Forcing Chains、v30仕様書の Deep Contradiction | **未実装（仕様のみ）** | **新規実装** |
| **予約セット（ホール定理）** | Queens Solver "Triple-Check"、数独 Naked Subset の一般化 | **未実装** | **新規実装** |
| 全解探索（バックトラック + MRV） | Queens Solver "Search" | 実装済 | ― |

## 3. v35 の新規アルゴリズム

### A. 連鎖伝播付き背理法 `_propagate_state` / `_state_broken`
- 候補マスに猫を仮定し、「基礎波及＋唯一マス確定」を不動点まで連鎖適用する。
- 破綻条件: 猫同士の競合 / 行・列・色のいずれかで猫を置けるマスが消滅 /
  k色の候補が k本未満の行（または列）にしか残らない（ホール条件。2色が同じ1行を奪い合う Double-Block 型もここで検出）。
- 破綻すれば仮定マスは X 確定。逆に、X と仮定して伝播が破綻すれば猫確定。
- 理由文には連鎖で確定した猫と破綻箇所を含め、人間が追える形で出力する。
  例:「R03C04 に猫を置くと（連鎖で R05C06 などが猫に確定）、色(ID:5)に猫を置けるマスがなくなるため矛盾する（連鎖伝播付き背理法）」

### B. 予約セット `_find_reserved_sets`
- k本の行（または列）の候補がちょうど k色に限られ、その色がその行（列）の外に置けない場合、
  その行（列）は k色で予約されるため、行内の他色マスを X に確定する（k=2,3）。

## 4. 検証結果
- サンプル盤面（sample_grid_colors, 8x8）: v34 では論理推論は 4ステップ（X 4マス）で停止し猫は 0 確定だったが、
  v35 では **35ステップで全8匹の猫を論理のみで確定** し、全解探索の一意解と一致した。
- 手法別内訳: 連鎖伝播付き背理法 22 / 唯一マス 8 / 限定配置 3 / 交差推論 2。
- pytest: 既存3件＋新規2件（完解テスト・理由文テスト）の計5件すべてパス。

## 5. 参考
- Queens Solver (https://storage.googleapis.com/queen_solver/queens_solver.html) の推論ルール解説
  （Region singleton / Elimination / Region forced row-col / Double-Block / Triple-Check / Lookahead / Search）
- 数独の解法テクニック体系（Hidden/Naked Single, Pointing, Claiming, Naked Subset, Forcing Chains）
