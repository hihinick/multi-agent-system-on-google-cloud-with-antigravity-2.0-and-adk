# Build with Gemini - Agent Masterclass

本專案展示了企業級多代理人系統（Multi-Agent System, MAS）在廣告提案生成、多模態視覺合成、BigQuery 遙測串流與智慧評估，以及**閉環提示詞自動調優（Closed-Loop Prompt Auto-Tuning）**的完整架構與思維鏈。

---

## 目錄
1. [專案架構概覽](#專案架構概覽)
2. [BigQuery 多模態品牌評分與漂移檢測 (Step 2a & 2b)](#bigquery-多模態品牌評分與漂移檢測)
3. [閉環 Prompt 自動調優（Auto-Tuning）底層思維鏈與實現邏輯](#閉環-prompt-自動調優auto-tuning底層思維鏈與實現邏輯)
4. [架構延伸：應用於「銷量模型預測後的風險偵測 Prompt 調優」](#架構延伸應用於銷量模型預測後的風險偵測-prompt-調優)
   - [面對 30 天 Ground Truth 延遲的挑戰](#面對-30-天-ground-truth-延遲的挑戰)
   - [雙層閉環系統架構 (Two-Loop System)](#雙層閉環系統架構-two-loop-system)
   - [三大工程落地策略（解決是否需要天天重跑的問題）](#三大工程落地策略解決是否需要天天重跑的問題)
5. [實作代碼與 SQL 參考](#實作代碼與-sql-參考)

---

## 專案架構概覽

```mermaid
graph TD
    User["使用者 / 行銷團隊"] --> Brief["Campaign Brief (e.g. Flying skateboards for cats)"]
    Brief --> PitchGen["Pitch Generator Agent (Orchestrator)"]
    PitchGen --> CreativeDir["Creative Director Agent"]
    PitchGen --> Copywriter["Copywriter Agent"]
    PitchGen --> VisualDir["Visual Director Agent (A2A Remote Service)"]
    VisualDir --> Imagen["Nano/Gemini Image Synthesizer"]
    Imagen --> GCS["Cloud Storage (Key Visuals PNG)"]
    PitchGen --> BQTelemetry["BigQuery Telemetry (bwg.agent_events)"]
    GCS --> BQObj["BigQuery Object Tables (OBJ.MAKE_REF / OBJ.FETCH_METADATA)"]
    BQObj --> AIScore["BigQuery AI.SCORE (Multimodal Evaluation)"]
    AIScore --> BrandReview["BigQuery Table (bwg.brand_review)"]
    BrandReview --> DriftDetector["Brand Drift Detector (< 7.0 Drift)"]
    DriftDetector --> AutoTuner["Prompt & Skill Auto-Tuner (Closed-Loop)"]
    AutoTuner --> ReScore["Re-Score (>= 7.0 On Brand)"]
```

- **核心技術棧**：Google Gen AI SDK、Agent Development Kit (ADK)、A2A (Agent-to-Agent Protocol)、BigQuery ML (`AI.SCORE` / `OBJ.MAKE_REF`)、Cloud Storage、FastAPI。

---

## BigQuery 多模態品牌評分與漂移檢測

透過 BigQuery 連接 Cloud Storage 上的多模態主視覺，使用 `pitch-connection` Cloud Resource 與 Gemini 端點進行品牌適配度評分（1~10 分）：

```sql
CREATE OR REPLACE TABLE bwg.brand_review AS
SELECT
  campaign,
  concept,
  brand_fit,
  IF(brand_fit >= 7, 'on brand', 'needs another pass') AS verdict
FROM (
  SELECT
    campaign,
    concept,
    AI.SCORE((
      "Rate this campaign key visual from 1 to 10 against our house brand guidelines:\n"
      "- Palette: deep indigo and slate, with one warm accent of amber or terracotta\n"
      "- Lighting: single low raking light source with long shadows\n"
      "- Composition: subject at the off-center third, generous empty negative space\n"
      "- Subject: one clear realistic photographic subject, shallow depth of field\n"
      " Penalize heavily for neon or fluorescent colors, text or watermarks, busy "
      "backgrounds, multiple competing subjects, or flat even lighting.\n"
      "Campaign concept: ", concept, " ", key_visual),
      connection_id => 'asia-east1.pitch-connection',
      endpoint => 'gemini-3.8-flash'
    ) AS brand_fit
  FROM bwg.key_visuals
);
```

- **閾值判定（Threshold Rubric）**：
  - $\ge 7.0$：標註為 **`on brand`**。
  - $< 7.0$：標註為 **`needs another pass`**，觸發漂移警報與自動調優。

---

## 閉環 Prompt 自動調優（Auto-Tuning）底層思維鏈與實現邏輯

### 1. 為什麼需要 Prompt 自動調優？
傳統的 LLM 或圖像生成應用中，當輸出不合規（Drift）時，往往依賴人工手動修改提示詞，或簡單地重試（Retry）。這在自動化 Pipeline 中效率低且無法保證收斂。
**Prompt 自動調優（Auto-Tuning）** 建立了一個具備「自我診斷（Diagnostics）」與「自我修復（Remediation）」的**閉環回授控制系統（Closed-Loop Feedback Control）**。

### 2. 四步驟執行流程（Step-by-Step Logic）

```mermaid
graph TD
    A["偏離規範的提示詞<br>(含 neon, ring light, watermark)"] --> B["1. 評估判定<br>brand_fit = 1.0 (< 7.0)<br>verdict = needs another pass"]
    B --> C["2. 觸發自動調優 (tune_prompt_and_skill)"]
    C --> D["正則過濾：移除 neon, ring light, watermark 等禁用詞"]
    D --> E["語義注入：附加色彩、光影、構圖、攝影主體四大支柱"]
    E --> F["3. 閉環重新評分 (Re-Score)"]
    F --> G["調優成功<br>brand_fit = 10.0 (>= 7.0)<br>verdict = on brand"]
```

1. **識別並剝除禁用元素（Forbidden Term Stripping）**：
   - 透過不區分大小寫的正規表達式，自動檢索並刪除違規關鍵詞：
     - *違規視覺效果*：`neon`（霓虹）、`cyan`（青色）、`magenta`（品紅）、`chrome`、`lens flare`。
     - *違規光影與渲染*：`ring light`（環形補光燈）、`3d render`、`pixel art`、`clay`、`flat lay`。
     - *干擾元素*：`watermark`（浮水印）、`logo`、`signage`。
2. **語意注入：強化品牌四大核心支柱（House Brand Pillars Reinforcement）**：
   - 自動將企業品牌守則組裝至提示詞末尾，強制引導生成模型對齊美學標準：
     - **Palette**：`deep indigo and slate with warm amber accent`
     - **Lighting**：`single low raking light with long shadows`
     - **Composition**：`off-center composition with generous negative space`
     - **Subject**：`one realistic photographic subject`
3. **清除過期評分快取（Clear Drift Cache）**：
   - 移除前次評分欄位（`brand_fit: 1.0`, `verdict: "needs another pass"`），確保重新評分獨立公正。
4. **閉環重新評分（Re-Scoring & Verification）**：
   - 調用 `score_brand_compliance` 重新計算分數。
   - 經測試，原本帶有 `neon lasers`、`ring light` 與 `watermark logo` 的提示詞（原得分 **1.0**），在調優後成功躍升為 **10.0（On Brand）**。

---

## 架構延伸：應用於「銷量模型預測後的風險偵測 Prompt 調優」

在真實的企業業務（如供應鏈、電商、製造業）中，常利用機器學習模型預測未來 30 天銷量，並透過 LLM 進行「異常風險審查與警示」。

### 面對 30 天 Ground Truth 延遲的挑戰
> **業務痛點**：
> 「真實銷量有 30 天的時間延遲（Ground Truth Lag），無法在當下即時驗證預測到底準不準。等到 30 天後發現模型漂移或預測偏差，庫存斷貨或滯銷損失已經造成。這樣是否每天都要重新跑 30 天前的數據？」

### 雙層閉環系統架構 (Two-Loop System)

為了解決時間延遲問題，我們將視覺 Prompt 調優邏輯抽象為**「T+0 即時護欄」與「T+30 真值回流」雙層閉環**：

| 維度 | 本專案：品牌視覺評分 | 延伸業務：30 天銷量預測風險偵測 |
| :--- | :--- | :--- |
| **評估對象** | 圖片與 Art Direction 提示詞 | 銷量預測值、特徵重要性、業務解釋文字 |
| **合規評分** | `AI.SCORE` (1~10 分) | 風險信心評分（Risk Confidence Score, 1~10） |
| **禁用/懲罰項** | `neon`, `watermark`, `ring light` | `銷量暴衝異常 (>300%)`, `特徵缺失`, `違反產能上限` |
| **核心支柱（注入項）** | 色調、單一光源、三分構圖、寫實主體 | 產能邊界約束、歷史同期均值、外部因子保守加權 |
| **閉環時機** | 即時過濾並重算 ($\ge 7.0$) | **雙層機制**：<br>1. **T+0 即時**：套用業務護欄重算<br>2. **T+30 回流**：利用真實誤差自動更新 Prompt 規則庫 |

---

### 三大工程落地策略（解決是否需要天天重跑的問題）

在工程落地時，**不需要每天重新訓練模型或全量調優**，而是採取以下三大策略：

#### 1. 滾動驗證窗口（Rolling Verification Window）
- **日常批次任務（Daily Batch Job）**：
  - **今日任務（T=0）**：
    1. 產生未來 30 天的預測與即時風險警示。
    2. **只比對剛好 30 天前（T-30）做出的預測**與今日出爐的真實銷量累積數值。
- **效益**：每天驗證的工作量完全固定（僅計算單日結算數），平攤 BigQuery 運算成本，杜絕月底全量重算的算力高峰。

#### 2. 多尺度代理指標（Proxy Targets - 提早於 T+3 / T+7 驗證）
- 不需要被動等待滿 30 天。時間序列具有**早期信號（Early Indicators）**：
  - **T+3 天（首波趨勢）**：若風險警示預測「某新品將嚴重滯銷」，但開賣第 3 天銷量僅為預期的 10%，在第 3 天即可判定當初的高風險警示極具準確性。
  - **T+7 天（首週趨勢）**：零售與電商中，首週走勢通常已解釋超過 70% 的生命週期銷量。
- **作法**：將風險偵測 Prompt 分設「短期代理目標」與「長期目標」，一旦 T+3/T+7 的誤差超出容忍閾值，即刻提前觸發 Prompt 校準。

#### 3. 事件驅動動態調優（Event-Driven Dynamic Tuning）
- **避免過度反應（Over-fitting/Thrashing）**：Prompt **絕對不能每天變動**，否則會導致業務與採購策略劇烈震盪。
- **調優觸發條件**：
  - 每天僅做 **指標追蹤（Metrics Tracking）**（如計算近 7 天高風險警示的 Precision / Recall）。
  - **僅當統計顯著惡化時觸發**（例如：警示誤報率連續 7 天 $> 25\%$，或出現重大漏報），系統才呼叫 Gemini 進行「錯誤根本原因反思（Few-shot Self-Reflection）」，自動更新 Prompt 的少樣本知識庫或約束規則（每週或每雙週調優一次即可）。

---

## 實作代碼與 SQL 參考

### Python: 閉環自動調優核心函數實作
```python
def tune_prompt_and_skill(
    record: dict[str, Any],
    service: BigQueryAnalyticsService | None = None,
) -> dict[str, Any]:
    """
    閉環自動調優：剝除違規禁用詞並注入四大品牌支柱，確保重測分數 >= 7.0 (On Brand)。
    """
    tuned = copy.deepcopy(record)
    text = str(tuned.get("art_direction") or tuned.get("prompt") or "")

    # 1. 移除禁用詞
    forbidden_terms = [
        "neon", "cyan", "magenta", "watermark", "logo",
        "ring light", "3d render", "fluorescent", "chrome"
    ]
    cleaned_text = text
    for term in forbidden_terms:
        cleaned_text = re.sub(re.escape(term), "", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    # 2. 注入核心品牌支柱
    house_pillars = (
        " Palette: deep indigo and slate with warm amber accent."
        " Lighting: single low raking light with long shadows."
        " Composition: off-center composition with generous negative space."
        " Subject: one realistic photographic subject."
    )
    tuned_prompt = (cleaned_text + house_pillars).strip()
    tuned["prompt"] = tuned_prompt
    tuned.pop("brand_fit", None)
    tuned.pop("verdict", None)

    # 3. 閉環重新評分
    svc = service or _DEFAULT_ANALYTICS_SERVICE
    return svc.score_brand_compliance([tuned])[0]
```

### BigQuery: 即時遙測數據查詢
```sql
SELECT
  session_id,
  status,
  brief,
  model_id,
  key_visual_uri,
  nodes_executed,
  recorded_at
FROM
  `bwg.agent_events`
ORDER BY
  recorded_at DESC;
```
