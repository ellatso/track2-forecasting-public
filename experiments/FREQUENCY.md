## Executive summary (read this first)

Run the second research round with separate daily and monthly selection. The daily
comparison changes volatility only. Monthly alternatives compare level reversion
and an attenuating trend with the unchanged local baseline. Keep all output outside
GitHub. This workflow does not build or upload a competition submission.

### Windows 一次執行

從研究分支更新後，在 PowerShell 執行：

```powershell
$repo = "C:\Users\ella.tso\Downloads\agenthon-t2"
git -C "$repo" pull --ff-only origin research/t2-experiments
powershell -ExecutionPolicy Bypass -File "$repo\experiments\run-frequency-windows.ps1" -Phase all
```

結果在 Downloads\Agenthon-T2-Research-V2\run-時間，與第一輪分開。
選參數、鎖定驗證依序執行；失敗就停止。Notebook 開啟後選 Run All Cells，
存檔才會保留顯示結果。Notebook 不重跑模型。

### 固定比較

- 日頻：300 筆歷史、漂移係數 1、原始相關性、1000 抽樣、種子 0/17/41。
  只比較樣本標準差與半衰期 60 筆的 EWMA（近期觀測權重較大）標準差。
- 月頻基準：原有 300 筆漂移 random walk（下一步等於上一期加變動）。
- 月頻水準回歸：最近 120 筆估計 AR(1)，即下一期由上一期水準線性推算。
  持續性係數限制在 0 到 0.98；每一步加入擬合殘差的相關隨機變動。
- 月頻阻尼趨勢：最近 12 筆線性趨勢，每前進一期乘 0.8，逐漸降低外推趨勢。
  變動標準差和相關性由最近最多 120 筆的一階變動估計。
- 模型每個案例只讀該卡截至歷史起點的資料，不查其他卡、不呼叫 House。

### 預先鎖定時段與限制

選參數起點為 2011/2013/2015/2017 年的指定日期；所有選參數目標必須早於
2018-04-01。新的驗證起點為 2018-04-02、2019-04-01，所有目標必須早於
2020-01-01。上一輪 2021/2023 年已讀過的結果不再用於選參數。

這些起點未用於第一輪鎖定驗證，但整段公開歷史先前被探索過，不能稱為完全未見
的正式 OOS。2018/2019 年也不能驗證疫情之後的月頻機制。所有模型每案例使用相同
抽樣數和種子，但不同動態模型產生路徑的方式不同。月頻覆蓋少時，不宣稱統計可靠。

兩個頻率各自按資產組合等權的選參數平均損失選出一個候選。平手保留基準。
驗證只比較已選候選與基準；不改挑第二名。程式核對設定、來源、輸入位元組、套件、
Python 版本及選參數表雜湊，拒絕重複驗證。已完成的選參數不能跨版本續跑。

### 如何看結果

每頻率各有 summary、cases、cells、by_family_seed CSV。

- summary：資產組合等權的本地相對損失；基準 1，越低越好。
- cases：邊際、聯合、尾端原始損失與基準值、損失比、加權貢獻及區間覆蓋。
  三項貢獻相加等於 composite。單一評分 cell 的聯合比與貢獻為 0。
- cells：每資產、原始 horizon key、真正觀測步數的邊際 CRPS，以及预测均值和區間。
  CRPS 使用主辦方共用實作；沒有另寫評分公式。
- validation.json：每頻率的候選、後段是否改善、資產組合重抽的描述性區間。
  資產組合可能共享工具，因此不能當獨立樣本推論。

本地以本地基準損失作分母，分母很小時損失比會放大。這不是官方 M0 正規化，
不能換算排行榜分數，也不能只憑後段有改善就直接升級提交。

不要公開上傳已執行 Notebook、逐筆結果、ZIP、Team Key 或 team-claim.json。
