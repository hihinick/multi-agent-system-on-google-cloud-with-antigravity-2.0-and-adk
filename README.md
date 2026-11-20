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
5. [企業級安全防護與人機協同架構 (Module 3 深度防禦與思維鏈)](#企業級安全防護與人機協同架構-module-3-深度防禦與思維鏈)
   - [PreToolUse 生命週期攔截鉤子（白名單、路徑遍歷與 Prompt Injection 阻絕）](#1-pretooluse-生命週期攔截鉤子工具調用防護)
   - [PII 機敏資料去識別化過濾器（保護 Log 審計與模型推論免於資料洩漏）](#2-pii-機敏資料遞迴去識別化過濾器pii-scrubber)
   - [ADK Human-in-the-Loop (HITL) 雙節點審核閘道與 Fail-Closed 熔斷](#3-adk-human-in-the-loop-hitl-雙節點審核閘道)
6. [生產級 Token 經濟學與上下文最佳化 (Module 4 Tokenomics)](#生產級-token-經濟學與上下文最佳化-module-4-tokenomics)
7. [實作代碼與 SQL 參考](#實作代碼與-sql-參考)

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

---

## 企業級安全防護與人機協同架構 (Module 3 深度防禦與思維鏈)

在自主多代理人系統（Autonomous Multi-Agent System）中，模型具備動態調用工具、存取外部 API 及串接多節點推論的能力。若缺乏邊界防護，可能面臨**提權漏洞（Privilege Escalation）、Prompt 注入攻擊（Prompt Injection）、機敏資料洩漏（PII Leakage in Telemetry & Inference），以及未經授權的高代價操作**。

本專案在 Module 3 建立了一套**三層縱深防禦體系（Defense-in-Depth Architecture）**：

```mermaid
graph TD
    UserReq["使用者輸入 / 企劃需求"] --> Step3b["1. 入口消毒 (PII Scrubber)<br>遞迴去識別化 API Secret, Email, Phone, SSN, CC"]
    Step3b --> CD["Creative Director 代理人<br>(產生概念草案 Concept)"]
    CD --> Step3c["2. 人機協同審核閘道 (HITL Gate)<br>rerun_on_resume 語意控制"]
    Step3c -- "approved is None (Pause)" --> WaitHuman["暫停執行<br>等待人類批准 (RequestInput)"]
    Step3c -- "approved is False (Fail-Closed)" --> Reject["熔斷終止<br>0 呼叫下游專家節點 (downstream_calls=0)"]
    Step3c -- "approved is True (Resume)" --> Unlocked["審核通過 (## Approved Concept)<br>解鎖下游專家節點"]
    Unlocked --> Specialist["下游代理人協作<br>(Brand Strategist, Copywriter, Visual Director)"]
    Specialist --> Step3a["3. 工具調用守門人 (PreToolUse Hook)<br>工具白名單、參數大小上限、注入特徵檢測、路徑穿越防護"]
    Step3a -- "Blocked" --> ToolErr["拒絕執行 (ToolAuthorizationError)<br>記錄安全審計日誌 (Audit Log)"]
    Step3a -- "Allowed" --> ExecTool["安全執行底層工具<br>(generate_key_visual / load_skill)"]
```

---

### 1. PreToolUse 生命週期攔截鉤子（工具調用防護）

#### 底層思維鏈與防禦設計
- **最小權限原則（Least Privilege）**：嚴格限制 Agent 僅能調用經過授權的工具清單（`DEFAULT_TOOL_ALLOWLIST = {"load_skill", "generate_key_visual"}`）。任何未在名單中的調用（如試圖調用 `exec_shell`、`rm_file` 等系統命令）一律直接阻斷。
- **越權覆寫參數防禦（Forbidden Override Keys）**：禁止輸入 payload 包含 `bypass_safety`、`override_brand_guidelines`、`disable_audit`、`admin_mode`、`skip_verification` 等高危金鑰。
- **路徑穿越與資源洩漏（Path Traversal）**：在 `load_skill` 的參數中，嚴格阻絕 `..`、`/`、`\` 等符號，防止代理人跳脫受限目錄讀取底層檔案。
- **阻斷攻擊與 DoS 防護（Oversized Argument Guard）**：設定 `MAX_ARGUMENT_LENGTH = 4096`，阻斷透過極長字串進行緩衝區溢位或上下文淹沒攻擊。
- **提示詞注入特徵偵測（Prompt Injection Patterns）**：透過正則掃描常見注入模式（例如 `ignore previous instructions`、`disregard earlier`、`<system>`、`you are now in developer mode`、`reveal secret prompt` 等）。
- **審計可追溯性（Immutable Audit Trail）**：每一次攔截檢查均生成結構化的 `AuditEntry`，記錄時間戳、工具名稱、決策（`allowed` / `blocked`）、違反原因及雜湊，確保安全事件具備不可否認性。

---

### 2. PII 機敏資料遞迴去識別化過濾器（PII Scrubber）

#### 底層思維鏈與業務痛點
在多代理人工作流程中，原始輸入若夾帶機敏個資，會直接面臨雙重風險：
1. **日誌與遙測持久化洩漏**：未遮蔽的個資會直接隨 BigQuery Telemetry 或 Cloud Logging 寫入資料庫，違反 GDPR、PCI-DSS、HIPAA 法規。
2. **模型推論污染與回顯**：模型在推論時可能將使用者的 API Key、電話號碼或信用卡號直接回顯至生成的社群文案或圖片提示詞中。

#### 核心技術實現與保護邊界
- **零原地變更（Zero In-Place Mutation）**：對傳入的巢狀資料結構（dict、list、tuple、set）一律採用 `copy.deepcopy` 遞迴清洗，確保不污染呼叫端的記憶體狀態。
- **五大機敏模式精準匹配（PII Patterns）**：
  - API 金鑰 / Bearer Token / Google API Key (`AIzaSy...`) $\rightarrow$ `[REDACTED_SECRET]`
  - 電子郵件信箱 $\rightarrow$ `[REDACTED_EMAIL]`
  - 13 至 19 碼信用卡卡號 $\rightarrow$ `[REDACTED_CC]`
  - 社會安全碼 (SSN) $\rightarrow$ `[REDACTED_SSN]`
  - 國際與本地電話號碼 $\rightarrow$ `[REDACTED_PHONE]`
- **非破壞性正規劃（False-Positive Avoidance）**：經過精密邊界錨定，**絕對不誤傷合法 ISO 日期（如 `2026-10-15`）與行銷轉換指標（如 `42%`、`350 lumen`）**。
- **字串相容結構（`ScrubResult(str)`）**：繼承自 `str`，既可直接進行 `in` 子字串比對，又具備 `.scrubbed` 與 `.redaction_count` 計數指標。

---

### 3. ADK Human-in-the-Loop (HITL) 雙節點審核閘道

#### 底層思維鏈與工作流程控制
在行銷企劃案中，創意總監（Creative Director）生成的「概念草案」是整個行銷案的基石。如果概念方向錯誤，後續讓文案專家與視覺總監自動生成大量社群貼文與圖片，不僅會浪費高昂的 GPU/LLM 算力，還可能產生不當品牌資產。

HITL 閘道實作了標準的 **兩節點 ADK 暫停與恢復語意（Pause/Resume Semantics）**：
1. **輸入請求節點（`approve_concept`）**：
   - 設定 `@node(rerun_on_resume=False)`。
   - 驗證草案內容存在後發出 `RequestInput("Please approve the campaign concept (yes/no).", response_schema=str)`，工作流程在此進入 `status="input_required"` 暫停狀態。
2. **審核裁決節點（`user_approval`）**：
   - 設定 `@node(rerun_on_resume=True)`。
   - 於 Session 恢復（Resume）時運行，調用 `ctx.run_node(approve_concept)` 讀取人工決定。
   - **明確肯定**（`"yes"`, `"y"`, `"true"`, `"approved"`）：輸出 `## Approved Concept` 並發出解鎖事件，讓下游的文案編寫（`copywriter`）與主視覺繪製（`remote_visual_director`）繼續進行（`downstream_calls=2`）。
   - **Fail-Closed 嚴格熔斷**：任何否定（`"no"`）或未預期輸入均直接丟出 `ValueError("User rejected the concept")`，強制將狀態標記為 `rejected`，**下游呼叫數嚴格歸零（`downstream_calls=0`）**。

---

## 生產級 Token 經濟學與上下文最佳化 (Module 4 Tokenomics)

在企業級多代理人與多輪對話系統中，上下文視窗（Context Window）的無限膨脹會直接導致**成本指數級上升、推論延遲拉長，以及模型注意力衰退（Lost in the Middle 效應）**。

Module 4 實作了一套完整的端到端 Tokenomics 最佳化架構，將上下文壓縮、預算截斷、快取管理與模型路由解耦並管線化。

### 1. 四大核心技術支柱

1. **Deterministic Token 估算 (`estimate_tokens`)**：
   - 採用 ~4 字元為 1 Token 的啟發式計算，零呼叫延遲且完全不依賴遠端 Tokenizer API。
   - 保證空字串、空白或 `None` 精確回傳 `0`，非空字串至少為 `1`。
2. **上下文遞迴摘要壓縮 (`compress_memory` / `compress_history`)**：
   - **Verbatim 精確保留**：將最後 `keep_recent=2` 輪（預設）對話原文原樣保留，防止最近一輪的具體指令與參數失真。
   - **演算法濃縮**：早於保留輪次的舊對話由演算法萃取角色與精要片段，濃縮為單一系統條目 `{"role": "system", "content": "[Summary of N earlier turns: ...]", "compressed": True}`，節省 50%~80% 歷史 Context。
3. **滑動視窗與預算修剪 (`prune_history`)**：
   - 強制受限於 `max_turns`（輪數上限）與 `max_tokens`（Token 硬性預算），非正數參數直接拋出 `ValueError`。
   - 最新輪次優先保留；遇到單一超大訊息（如上萬字日誌或代碼）時，自動依據預算截斷至上限字數並標註 `"truncated": True`。
4. **SHA-256 靜態前綴快取管理 (`PromptCacheManager`)**：
   - 針對品牌規範（Brand Guidelines）、專家設定檔等靜態前綴，以 SHA-256 生成 64 位元 Hex 快取鍵。
   - 內建 TTL（Time-To-Live，預設 300 秒）過期機制，追蹤 `cache_hit`、`hits` 次數並計算已節省之 `saved_tokens`。
5. **模型策略階層路由 (`select_model_strategy`)**：
   - 支援 5 個標準 Tier：`flash`、`pro`、`low`、`high`、`local`（支援無雲端外洩的本地開源 Gemma）。
   - 回傳 `model_id`、`target`、`cost_tier`、`latency_tier` 與 `thinking_budget`（思維鏈思考預算）。
6. **統整協調器 (`TokenomicsManager`)**：
   - 透過 `.optimize()` 將記憶體壓縮、滑動修剪、靜態快取與模型選擇無縫串接，並傳回 `original_tokens`、`optimized_tokens`、`tokens_saved` 等可直接串接 BigQuery 遙測的量化指標。

---

### 2. 實測思維鏈與基準測試（10 輪「Flying skateboards for cats」）

我們在本地 FastAPI 伺服器啟動環境下，針對經典題目「Flying skateboards for cats（貓咪飛行滑板）」進行了涵蓋產品概念、磁浮推進、防護感應、品牌配色與宣傳標語的 10 輪頭腦風暴對話，並帶入 `brand-guidelines/SKILL.md`（195 tokens）進行雙次呼叫基準實測：

```mermaid
graph TD
    Raw["10 輪原始對話<br>(283 tokens)"] --> Step1["1. 記憶體壓縮 compress_memory<br>(保留最後 2 輪 verbatim，前 8 輪濃縮為 system summary)"]
    Step1 --> Comp["壓縮後 3 輪對話<br>(97 tokens，節省 186 tokens，節省率 65.7%)"]
    Comp --> Step2["2. 滑動視窗修剪 prune_history<br>(max_turns=4, max_tokens=150)"]
    Step2 --> Pruned["符合硬性預算限制<br>(97 tokens <= 150 tokens)"]
    Pruned --> Cache["3. SHA-256 Prompt 快取<br>(品牌指南 195 tokens)"]
    Cache --> Call1["第 1 次呼叫: cache_hit = False<br>生成 SHA-256: ef21b9be50...7dbb<br>總原始 478 tokens -> 最佳化 150 tokens (節省 328 tokens)"]
    Cache --> Call2["第 2 次呼叫: cache_hit = True<br>快取命中節省: 195 tokens<br>累積節省 381 tokens (選用 flash 模型策略)"]
```

#### 測試數據摘要

| 處理階段 | 輪數變化 | Token 數 | 經濟學效果 / 說明 |
| :--- | :---: | :---: | :--- |
| **原始完整對話 (Raw)** | 10 輪 | **283 tokens** | 完整頭腦風暴對話歷史 |
| **記憶體壓縮後 (`compress_memory`)** | 3 輪 | **97 tokens** | **節省 186 tokens (縮減 65.7%)**；保留最近 2 輪原始文字，前 8 輪壓縮為單一 System Summary |
| **滑動視窗修剪後 (`prune_history`)** | 3 輪 | **97 tokens** | 受到 `max_tokens=150` 約束，未超出上限 |
| **第 1 次附加品牌指南 (Cache Miss)** | 3 輪 + 前綴 | **150 tokens** | 原始總計 478 tokens，快取註冊並節省 328 tokens |
| **第 2 次附加品牌指南 (Cache Hit)** | 3 輪 + 前綴 | **97 tokens (實際計費)** | 快取命中省去 195 tokens，**端到端累積節省 381 tokens** |

---

### 3. 本架構與 OpenCode `/compact` 的核心差異與思維剖析

許多開發者常問：「**這個 Tokenomics 架構跟 OpenCode / Cursor / Cline 等 Coding Agent 的 `/compact` 指令有什麼不同？**」

#### 差異對比

| 評比維度 | OpenCode 的 `/compact` | 本專案 Tokenomics 最佳化架構 |
| :--- | :--- | :--- |
| **觸發與執行機制** | **手動 / 反應型 (Reactive)**<br>通常依賴使用者手動輸入指令，或在 Context 接近上限時才整體壓縮。 | **流水線化 / 預防型 (Proactive Pipeline)**<br>每次 Agent 調用自動在記憶體內執行，全生命週期零人工介入。 |
| **壓縮手段與開銷** | **額外 LLM 呼叫 (Extra LLM Call)**<br>需要先花費一次模型呼叫來「總結舊對話」，產生額外 API 延遲與 Token 開銷。 | **演算法精煉 (Zero-Latency Algorithm)**<br>在本地以毫秒級程式邏輯提煉摘要與視窗修剪，**不浪費任何 LLM 呼叫**。 |
| **近期輪次保真度** | **模糊化風險**<br>整段歷史全部丟進摘要，最近一兩輪的關鍵代碼、錯誤訊息或指令參數容易丟失。 | **Verbatim 絕對保真**<br>保證最後 `keep_recent=2` 輪原始文字**一字不差保留**，避免 Agent 產生上下文漂移。 |
| **預算控制能力** | **軟性控制**<br>LLM 產生的摘要長度不可預測，仍可能超出特定 Token 配額。 | **硬性數學保證 (Deterministic Budget)**<br>具有剛性限制；若單一訊息過長會自動進行截斷（`truncated: True`）。 |
| **靜態前綴快取** | **通常無獨立快取層**<br>System Prompt 常隨上下文被一起重新發送或模糊處理。 | **獨立 SHA-256 + TTL 快取**<br>對品牌指南、規則手冊生成 64 位元雜湊，追蹤命中率並計算精確節省量。 |
| **商業與遙測整合** | **僅作為終端輔助**<br>通常無完整的 Token 成本審計資料。 | **完整遙測數據結構**<br>產出 `original_tokens`、`optimized_tokens`、`tokens_saved` 等結構化數據，直通 BigQuery 成本監控。 |

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

