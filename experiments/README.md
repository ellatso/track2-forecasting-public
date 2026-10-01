# Executive summary (read this first)

這是離線研究分支，不是新的提交容器。20組數值設定共用四個選參數日期、三個種子與1000個樣本；選定一個候選後，才在兩個較晚日期比較該候選與固定基準。Jupyter 只讀結果。所有結果、環境和已執行 Notebook 放在 Downloads，不能推上公開 repository。

## Windows：依序執行

```powershell
$repo = "C:\Users\ella.tso\Downloads\agenthon-t2"
git -C "$repo" fetch origin
git -C "$repo" switch research/t2-experiments
git -C "$repo" pull --ff-only origin research/t2-experiments
powershell -ExecutionPolicy Bypass -File "$repo\experiments\run-windows.ps1" -Phase select
```

完成後再跑鎖定候選的後期測試：

```powershell
powershell -ExecutionPolicy Bypass -File "$repo\experiments\run-windows.ps1" -Phase holdout
```

最後開 Notebook 看結果：

```powershell
powershell -ExecutionPolicy Bypass -File "$repo\experiments\run-windows.ps1" -Phase notebook
```

需要 Python Launcher 的 Python 3.13。程式建立獨立環境。`latest-run.txt` 記住最近成功的選參數目錄；也可傳 `-RunDir "完整路徑"` 指定另一個已存在的實驗。重跑選參數會建立新目錄；後期測試只能啟動一次，改過程式、輸入、環境或計畫會拒絕。

## 實驗設計

- 18組固定設定：窗口60／120／300觀測 × 漂移0／0.5／1 × 標準差倍率0.85／1。
- 另加目前 calibrated 的類別策略，以及60步半衰期的 EWMA 波動估計。
- 選參數日期：2011、2013、2015、2017年初。後期測試：2021、2023年初。
- 種子0／17／41，所有方法的種子與抽樣數相同；不依種子挑最好結果。
- 僅使用每張卡自己提供的資料。相同資產、期限、步數、分類的卡只取一張最長歷史，絕不拼接不同卡的答案。
- 每個歷史原點以前的資料用於預測；該序列之後、但仍在原卡截止日前的觀測只用於離線評估。
- 日資料至少300觀測，月資料至少60觀測；300觀測估計區間與未來區間若跨資料缺口則排除。月資料沿用官方解析的每資產觀測步數及發布延遲，不能把 horizon key 當月數。
- 所有候選使用完全相同的可評估案例。任何方法失敗會排除該案例的全部方法，並寫明原因。
- CRPS、joint 與 tail 直接呼叫官方模組，包括單一格點的有效權重。分母是同一歷史案例的本地300觀測漂移方法，**不是私有 M0，也不是正式排行榜分數**。
- 先對同一資產組合的日期、期限、種子取平均，再給每個資產組合相同權重。組合可能共享資產；bootstrap 只呈現描述性不確定性。

## 輸出與限制

`plan.json` 保存參數、日期、程式和輸入 SHA-256、Python／套件版本及 git 狀態。`selected.json` 保存只由選參數階段選出的候選。CSV 提供逐案例、分類、月／日頻率和種子診斷；`coverage.json` 明列不能評估的情況。後期只輸出選定候選和基準，不讓20組一起偷看測試資料。

這些公開歷史在先前探索中已被看過，因此是本次固定的後期測試，不能宣稱從未看過的獨立 OOS。此流程也不是官方 Final。普通歷史原點不等同於 F4 的衝擊事件；有缺口的轉移資產、過早的卡和無可用後期資料的卡可能無法評估。月資料缺少完整歷史 vintage 時，不能驗證當時的即時發布資訊。Notebook 會顯示覆蓋範圍，不能把未覆蓋部分當成功。

程式不呼叫 House：沒有歷史原點相應的凍結文本與正式憑證，不能假裝驗證文字增益。日後文字開／關提交必須使用相同基準、種子、抽樣數。此分支不變更已發布映像，也不自動打包或上傳。

## 核對真的上傳哪一版

只讀 ZIP 的 submission.json，輸出映像 digest 與模型名稱；不讀 team-claim.json、不顯示 Team Key／tokens／team_id：

```powershell
$python = "$env:USERPROFILE\Downloads\Agenthon-T2-Research\.venv\Scripts\python.exe"
& $python "$repo\experiments\audit_submission.py" "$env:USERPROFILE\Downloads\Agenthon-T2-609-calibrated\submission.zip"
& $python "$repo\experiments\audit_submission.py" "$env:USERPROFILE\Downloads\Agenthon-T2-609-integrated\submission.zip"
```

將輸出 digest 對照 `versions.json`，再記錄 Submission ID 與分數。目前 #954551／#954601 的分數已知，但 ZIP 對應版本仍待這項核對；不能依提交時間猜。
