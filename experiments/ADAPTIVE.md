## Executive summary (read this first)

Compare an unchanged daily baseline, the third-round fixed mixture, and a mixture
selected from each case's own pre-origin history. All outer periods have already
been explored, so this is a historical diagnostic, not independent validation.
Monthly forecasts are unchanged. The workflow creates no competition submission.

### Windows 重現

```powershell
$repo = "C:\Users\ella.tso\Downloads\agenthon-t2"
git -C "$repo" pull --ff-only origin research/t2-experiments
powershell -ExecutionPolicy Bypass -File "$repo\experiments\run-adaptive-windows.ps1" -Phase all
```

結果寫入 Downloads\Agenthon-T2-Research-V4\run-時間。
Notebook 開啟後 Run All Cells，再存檔。Notebook 只讀已完成結果。
也可用 `-Phase benchmark` 只跑比較，或 `-Phase notebook` 開啟最新結果。
重新跑比較需使用新資料夾；舊結果不覆寫。研究程式更新後，舊版尚未完成的
holdout 不能跨版本續跑，因為來源雜湊不同。

### 三種固定方法

1. 原基準：最多 300 筆歷史、漂移係數 1、普通樣本變異數、原始相關性。
2. 固定混合：普通變異數與近期 EWMA 變異數各占 50%，相關性收縮 25%。
3. 動態混合：每個案例在預測起點前完成內部驗證，再選近期變異數的權重。
   相關性仍收縮 25%，與固定混合控制一致。資料不足時回退至近期權重 0%，
   保留相同相關性控制；不是偷偷刪除難案例。

三者的漂移、300 筆窗口、期限累積、1000 抽樣與種子 0/17/41 都固定。
所有模型在同一組可用案例上比較。月頻不參與這輪。

### 動態權重如何選

- 內部候選權重：0%、12.5%、25%、37.5%、50%，事先固定。
- 每案例最多 6 個內部起點，間隔 21 個觀測。每個起點都保留自己的前 300 筆
  訓練資料，並預測外部案例所要求的同樣期限。
- 內部最長期限的目標必須完整落在外部起點已知的歷史內。
- 每候選使用 256 抽樣與共同種子 71。每個資產／期限的 CRPS 調用主辦方共用
  實作，再以該內部起點以前估出的標準差及期限尺度正規化，避免大單位資產支配。
- 平均內部損失最小者入選；數值平手選較小權重。內部不足 3 個案例時回退到 0%。
- 權重選擇不使用外部目標，也不隨外部抽樣種子改變。完全相同的數值輸入可重用
  計算快取，快取不以卡片 ID 或其他卡片資料做查詢。
- 內部案例的目標可能重疊，不能把 6 個案例當成獨立樣本。
  內部選擇目標只看邊際 CRPS；外部仍檢查邊際、聯合與尾端損失。

### 時段與限制

外部起點為 2011/2013/2015/2017 年先前起點，以及 2018/2019/2020/2021/2022/2023
年先前起點。每案例目標限制在起點所在年度，過長期限或缺資料會列出原因。
這項共同限制與前幾輪的部分範圍不同，因此不能直接逐年對照前輪未對齊的總表。

這些歷史以前已讀過，而且影響了本輪設計。全程預測只用起點以前資料，
不等於整個研究具有完全未見的 holdout。此輪不挑選新的後段冠軍、不宣稱官方改善。

### 結果與時序稽核

- diagnostic_summary.csv：資產組合等權的總體本地損失比。
- diagnostic_by_year.csv：每年重新按可用資產組合等權平均。每年覆蓋不同。
- diagnostic_cases.csv：三項評分的原始值、基準值、比值、加權貢獻與預測區間覆蓋。
- diagnostic_cells.csv：每資產／期限的邊際 CRPS、預測均值與區間。
- weight_audit.csv：每個案例的選定權重、內部案例數、末筆內部目標位置、
  可用歷史末筆位置、候選損失與歷史雜湊。目標位置不可大於歷史末筆位置。
- paired_diagnostics.json：依資產組合重抽的描述性差異。組合共享工具，不能據此
  宣稱獨立樣本統計顯著。
- plan.json：方法、日期、種子、套件、Python、程式及輸入雜湊，在跑分前寫出。

輸出留在公開 repo 之外；不要上傳已執行 Notebook、逐筆結果或任何 Team 認證資料。
