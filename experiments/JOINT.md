## Executive summary (read this first)

The third research round compares 12 daily path models. It varies the variance
mixture, correlation shrinkage, and long-horizon variance accumulation. Drift,
history window, seeds and draws stay fixed. Monthly forecasts remain at the local
baseline. This is historical diagnosis, not an official score or a submission.

### Windows 重現

```powershell
$repo = "C:\Users\ella.tso\Downloads\agenthon-t2"
git -C "$repo" pull --ff-only origin research/t2-experiments
powershell -ExecutionPolicy Bypass -File "$repo\experiments\run-joint-windows.ps1" -Phase all
```

結果寫入 Downloads\Agenthon-T2-Research-V3\run-時間。Notebook 開啟後 Run All Cells
再存檔。更新研究程式後不能續跑舊版未完成的 holdout，因為來源雜湊改變。

### 事先固定的比較

全部使用最多 300 筆歷史、漂移係數 1、1000 抽樣、種子 0/17/41。
每個預測只用自己卡片截至起點的資料。

- 波動混合：近期 EWMA 變異數的權重為 0、25%、50%，其餘使用普通樣本變異數。
  混合變異數後取平方根；不是直接平均標準差。
- 相關性收縮：普通相關矩陣的非對角元素乘 1 或 0.75，對角保留 1。
  只有一個資產時，這個控制不改變路徑。
- 長期限變異數累積：前 20 個商業日維持原有累積；之後比較線性累積或
  `20 * (h/20)^0.8`。所有期限共用一條隨機路徑，因此仍有一致的跨期限關係。
  這是待驗證的預測假設；沒有證據時不宣称它符合真實市場。

共 3×2×2＝12 組，包含原基準。選參數表可以拆開單一控制的效果；
後段只評估選定候選與基準，不替其他組重排後段名次。

### 選參數與驗證

選參數仍使用 2011/2013/2015/2017 年固定起點，全部目標須早於 2018-04-01。
第三輪新驗證起點固定為 2020-01-02、2022-01-03；目標必須分別早於
2021-01-01、2023-01-01。過長期限或缺資料的案例會列出排除理由。

這兩個起點未作為前兩輪驗證起點，但整段公開歷史以前已探索過，不能稱為完全未見
的正式 OOS。前兩輪結果已影響模型設計，這輪也沒有評估 House 文字增益。

按資產組合等權的前段平均本地損失選一組，平手保留基準。設定、來源、資料、
Python 與套件版本都有雜湊鎖；拒絕重複 holdout。後段改善只是繼續研究的線索，
不能直接換算官方排行榜改善，或宣稱冠軍。

### 結果輸出

summary 為資產組合等權結果。cases 保留各評分項目的原始值、基準值、損失比與
加權貢獻。cells 保留每資產與原始期限 key 的 CRPS、預測均值與區間。
validation.json 保留選定候選的後段差異與描述性不確定區間。

單一評分 cell 的聯合權重為零，其他權重會重分配，因此必須與基準的實際貢獻比，
不能假設每筆都是 50/30/20。Notebook 以資產組合等權顯示三项貢獻。

請保留結果在 repo 外，不公開已執行 Notebook、逐筆結果或任何提交認證資料。
