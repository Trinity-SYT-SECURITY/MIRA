# MIRA 專案技術總結

## 專案概述

**MIRA (Mechanistic Interpretability Research & Attack Framework)** 是一個用於分析和攻擊 LLM 安全機制的研究框架，使用機械可解釋性技術。

## 核心技術棧

### 1. **深度學習框架**
- **PyTorch** (≥2.0.0) - 核心深度學習框架
- **Transformers** (≥4.30.0) - HuggingFace 模型載入和推理
- **NumPy** (≥1.24.0) - 數值計算

### 2. **機器學習工具**
- **scikit-learn** (≥1.3.0) - PCA、分類器等機器學習工具

### 3. **數據處理與可視化**
- **Pandas** (≥2.0.0) - 數據處理
- **Matplotlib** (≥3.7.0) - 靜態圖表
- **Plotly** (≥5.15.0) - 交互式可視化
- **Seaborn** (≥0.12.0) - 統計圖表

### 4. **Web 可視化**
- **Flask** (≥3.0.0) - 實時 Web 儀表板
- **Flask-CORS** (≥4.0.0) - 跨域支持

### 5. **工具庫**
- **PyYAML** (≥6.0) - 配置文件解析
- **tqdm** (≥4.65.0) - 進度條
- **psutil** - 系統資源監控

## 專案架構

```
mira/
├── core/           # 模型包裝器、Hook 管理、配置
├── analysis/       # 子空間分析、激活分析、注意力分析
├── attack/         # 梯度攻擊、重路由攻擊、GCG、探針
├── metrics/        # ASR、距離、概率指標
├── visualization/  # 圖表、HTML 報告、流程可視化
└── utils/          # 環境檢測、日誌、數據工具
```

## 主要功能

### 1. **子空間分析**
- 在激活空間中尋找拒絕/接受方向
- 使用 PCA 進行降維
- 訓練線性探針識別安全機制

### 2. **攻擊方法**
- **梯度攻擊** - GCG 風格的對抗性後綴優化
- **重路由攻擊** - 將激活引導遠離拒絕方向
- **攻擊探針** - 19 種不同的攻擊（越獄、編碼、注入、社交）

### 3. **可視化**
- 層級流程追蹤
- 注意力熱圖
- 交互式 HTML 報告
- 實時 Web 儀表板（Flask）

## .env 配置說明

### 必需配置
**無** - 框架可以在沒有任何環境變數的情況下運行

### 可選配置

#### 1. **模型選擇**（跳過交互式選擇）
```bash
MODEL_NAME=EleutherAI/pythia-70m
```

#### 2. **HuggingFace 配置**
```bash
# 自定義緩存目錄（可選）
HF_CACHE_DIR=/path/to/custom/cache

# HuggingFace Token（僅用於受限模型如 Llama 2）
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

#### 3. **設備配置**（自動檢測）
```bash
DEVICE=auto          # auto, cuda, cpu, mps
DTYPE=float32        # float32, float16, bfloat16
```

#### 4. **輸出配置**
```bash
OUTPUT_DIR=./results
LOG_LEVEL=INFO
LOG_FILE=./logs/mira.log
```

## 推薦模型

### CPU 友好（2-4GB RAM）
- `EleutherAI/pythia-70m` - 最快，適合快速測試
- `EleutherAI/pythia-160m` - 小而強大
- `gpt2` - 經典基準模型

### GPU 推薦（4-8GB VRAM）
- `EleutherAI/pythia-1b` - 研究標準
- `EleutherAI/pythia-1.4b` - GPU 最佳平衡
- `EleutherAI/gpt-neo-1.3B` - 替代架構

### 大型模型（8GB+ VRAM）
- `EleutherAI/pythia-2.8b`
- `EleutherAI/gpt-j-6b`

### 對話模型（安全對齊，更難攻擊）
- `meta-llama/Llama-2-7b-chat-hf` ⚠️ 需要 HF_TOKEN
- `mistralai/Mistral-7B-Instruct-v0.2`

## 快速開始

### 1. 安裝依賴
```bash
pip install -e .
# 或
pip install -r requirements.txt
```

### 2. 運行完整流程
```bash
python main.py
```

### 3. 使用特定模型
```bash
# 方法 1: 設置環境變數
export MODEL_NAME=EleutherAI/pythia-70m
python main.py

# 方法 2: 創建 .env 文件
echo "MODEL_NAME=EleutherAI/pythia-70m" > .env
python main.py
```

### 4. 運行研究流程
```bash
python examples/run_research.py --model pythia-70m --output ./results
```

## 配置文件

### config.yaml
主要配置文件，包含：
- 模型設置（名稱、設備、數據類型）
- 分析配置（PCA 組件、批次大小）
- 攻擊配置（步數、學習率、後綴長度）
- 評估配置（成功閾值、拒絕模式）
- 可視化配置（顏色方案、DPI）

### .env（可選）
環境變數覆蓋，主要用於：
- 跳過交互式模型選擇
- HuggingFace 認證
- 自定義緩存位置

## 輸出結構

```
results/run_YYYYMMDD_HHMMSS/
├── charts/
│   ├── subspace.png          # 拒絕子空間可視化
│   └── asr.png               # 攻擊成功率
├── html/
│   └── mira_report.html      # 交互式報告
├── data/
│   ├── records.csv           # 所有攻擊記錄
│   └── records.json          # 詳細 JSON 記錄
└── summary.json              # 實驗摘要
```

## 重要注意事項

1. **無需 .env 文件** - 框架會自動檢測系統並推薦合適的模型
2. **自動 GPU 檢測** - 自動使用可用的 CUDA/MPS/CPU
3. **HF_TOKEN 僅在需要時使用** - 只有受限模型（如 Llama 2）才需要
4. **配置優先級** - .env > config.yaml > 默認值

## 獲取 HuggingFace Token

如果需要使用受限模型：
1. 訪問 https://huggingface.co/settings/tokens
2. 創建新 token（read 權限即可）
3. 在模型頁面接受使用條款
4. 將 token 添加到 .env: `HF_TOKEN=hf_xxxxx`

## 常見使用場景

### 場景 1: 快速測試（無需配置）
```bash
python main.py
# 系統會自動選擇合適的模型
```

### 場景 2: 使用特定模型
```bash
# 創建 .env
echo "MODEL_NAME=EleutherAI/pythia-1b" > .env
python main.py
```

### 場景 3: 使用 Llama 2（需要認證）
```bash
# .env 文件
MODEL_NAME=meta-llama/Llama-2-7b-chat-hf
HF_TOKEN=hf_your_token_here
```

### 場景 4: 自定義緩存位置
```bash
# .env 文件
HF_CACHE_DIR=/mnt/large_drive/huggingface_cache
```

## 總結

MIRA 是一個**開箱即用**的框架：
- ✅ 無需環境變數即可運行
- ✅ 自動檢測硬件並推薦模型
- ✅ 提供交互式模型選擇
- ✅ 生成完整的研究報告和可視化
- ⚠️ 僅在使用受限模型時需要 HF_TOKEN
