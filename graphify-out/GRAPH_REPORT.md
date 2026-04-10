# Graph Report - .  (2026-04-10)

## Corpus Check
- Large corpus: 252 files · ~59,468 words. Semantic extraction will be expensive (many Claude tokens). Consider running on a subfolder, or use --no-semantic to run AST-only.

## Summary
- 994 nodes · 1358 edges · 133 communities detected
- Extraction: 79% EXTRACTED · 21% INFERRED · 0% AMBIGUOUS · INFERRED: 280 edges (avg confidence: 0.51)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `TriTransformerModel` - 20 edges
2. `TestTriTransformerTrainer` - 19 edges
3. `ITransformer` - 18 edges
4. `TriTransformerTrainer` - 18 edges
5. `TestTriTransformerModel` - 18 edges
6. `BaseLoss` - 18 edges
7. `CTransformer` - 16 edges
8. `OTransformer` - 16 edges
9. `TriTransformerConfig` - 16 edges
10. `PositionalEncoding` - 14 edges

## Surprising Connections (you probably didn't know these)
- `HallucinationGuard (C-RAG based)` --conceptually_related_to--> `FactChecker`  [INFERRED]
  docs/tech_details/14_rag.md → docs/tasks/tri-transformer-backend-v2/tech-solution.md
- `Chunked Prefill` --semantically_similar_to--> `Chunking and Pooling`  [INFERRED] [semantically similar]
  docs/tech_details/20_vllm_pagedattention.md → docs/tech_details/02_chunking_pooling.md
- `SemanticSplitter Node Parser` --semantically_similar_to--> `Chunking and Pooling`  [INFERRED] [semantically similar]
  docs/tech_details/16_llamaindex.md → docs/tech_details/02_chunking_pooling.md
- `RAG Recall@5 Metric (1.0)` --rationale_for--> `Async Real-Time Retrieval Loop`  [INFERRED]
  eval/data/eval_report.md → docs/sub_prds/03_realtime_multimodal_rag_and_hallucination_control.md
- `Hallucination Rate Metric (0.0)` --rationale_for--> `Hallucination Blocking via C-Transformer`  [INFERRED]
  eval/data/eval_report.md → docs/sub_prds/03_realtime_multimodal_rag_and_hallucination_control.md

## Hyperedges (group relationships)
- **Tri-Transformer Three-Branch Tightly Coupled Closed Loop (I→C→O→C)** — prd_itransformer, prd_ctransformer, prd_otransformer [EXTRACTED 1.00]
- **Full RAG Pipeline Stack** — backend_rag_pipeline, tech_llamaindex, requirements_chromadb, backend_bge_embedder, backend_bge_reranker, requirements_rank_bm25 [EXTRACTED 1.00]
- **PyTorch Model Skeleton Implementation** — pytorch_branches_py, pytorch_tri_transformer_py, pytorch_trainer_py [EXTRACTED 1.00]
- **FR-101 Dual-End Pluggable LLM Implementation** — backend_v2_fr101, backend_v2_hf_model_loader, backend_v2_pluggable_llm_adapter, backend_v2_lora_adapter [EXTRACTED 1.00]
- **FR-102 Multimodal Tokenizer Stack** — backend_v2_fr102, backend_v2_text_tokenizer, backend_v2_audio_tokenizer, backend_v2_vision_tokenizer, backend_v2_unified_tokenizer [EXTRACTED 1.00]
- **FR-103 Streaming Inference Stack** — backend_v2_fr103, backend_v2_streaming_engine, backend_v2_websocket_route [EXTRACTED 1.00]
- **FR-104 Hallucination Blocking Stack** — backend_v2_fr104, backend_v2_fact_checker, backend_v2_hallucination_detected_field [EXTRACTED 1.00]
- **Tri-Transformer C-Transformer Control Mechanism** — tech_adaln_zero, tech_dit, tech_state_slots, tri_transformer_ctransformer [EXTRACTED 1.00]
- **Tri-Transformer I-C-O Closed Loop** — tri_transformer_itransformer, tri_transformer_ctransformer, tri_transformer_otransformer, tech_state_slots_closed_loop [EXTRACTED 1.00]
- **Multimodal Unified Token Space Design** — tech_anygpt_vocab, backend_v2_unified_tokenizer, tech_chameleon_early_fusion, backend_v2_fr102 [INFERRED 0.85]
- **Frontend v2 WebRTC Feature Set** — frontend_v2_webrtc_types, frontend_v2_webrtc_api, frontend_v2_webrtc_store, frontend_v2_webrtc_controls, frontend_v2_audio_visualizer [EXTRACTED 1.00]
- **LoRA/QLoRA Parameter-Efficient Fine-Tuning Techniques** — tech_lora, tech_qlora, tech_lora_nf4, tech_lora_peft_lib [EXTRACTED 1.00]
- **Real-Time RAG Knowledge Injection Pipeline** — milvus_db, llamaindex_framework, prd01_o_transformer [EXTRACTED 1.00]
- **Multimodal Unified Discrete Token Space** — bpe_multimodal_vocab, encodec_rvq, vqgan_vq_layer [EXTRACTED 1.00]
- **Distributed Joint Training Stack** — deepspeed_zero3, prd04_training_pipeline, prd01_lora_adapter [EXTRACTED 1.00]

## Communities

### Community 0 - "Loss & Metrics System"
Cohesion: 0.04
Nodes (26): BaseLoss, BaseLoss, ContrastiveControlLoss, ControlAlignmentLoss, InstructionFollowingLoss, KnowledgeConsistencyLoss, AbstentionCalibrationLoss, FactualHallucinationLoss (+18 more)

### Community 1 - "Bidirectional Encoder & ModernBERT"
Cohesion: 0.03
Nodes (93): adaLN-Zero Modulation in Encoder, BERT, Bidirectional Encoder, i_enc Semantic Representation, ModernBERT, BPE (Byte Pair Encoding), Multimodal Unified Token Space, SentencePiece Tokenizer (+85 more)

### Community 2 - "Frontend API & React Components"
Cohesion: 0.03
Nodes (7): handleKeyDown(), handleSend(), downloadConversation(), exportToJSON(), exportToMarkdown(), handleChange(), isValidFile()

### Community 3 - "Tri-Transformer Branches (I/C/O)"
Cohesion: 0.06
Nodes (23): CTransformer, ITransformer, OTransformer, PositionalEncoding, 输出解码器：受控自回归生成，融合 I 分支编码与 C 分支控制信号。, 输入编码器：将 token ids 编码为上下文感知的隐状态序列。, 控制中枢：通过双向交叉注意力监控 I/O 状态，输出全局控制信号。, model() (+15 more)

### Community 4 - "adaLN-Zero & DiT Control"
Cohesion: 0.03
Nodes (74): Diffusion Forcing (2024), FiLM (Feature-wise Linear Modulation), adaLN-Zero (Adaptive LayerNorm with Zero Initialization), adaLN-Zero in C-Transformer for Real-Time Control, adaLN-Zero: scale/shift/gate Modulation Mechanism, Rationale: Zero Initialization Prevents Gradient Explosion, AnyGPT (arXiv:2402.12226), EnCodec (+66 more)

### Community 5 - "Consistency & Evaluation Checks"
Cohesion: 0.06
Nodes (17): ConsistencyChecker, DocumentQAGenerator, DualModelValidator, Enum, GTFusionEngine, GTVersioning, KGTripleExtractor, GroundTruthDataset (+9 more)

### Community 6 - "CI Gate & Quality Pipeline"
Cohesion: 0.06
Nodes (16): CIGate, DialogCohesionEvaluator, EvalPipeline, _compute_bert_score_f1_approx(), _compute_bleu(), _compute_rouge_l(), GenerationEvaluator, _tokenize() (+8 more)

### Community 7 - "Chat Service & Inference"
Cohesion: 0.06
Nodes (15): forward(), ChatService, ChatMessage, ChatSession, Base, DeclarativeBase, OptionalHTTPBearer, Document (+7 more)

### Community 8 - "Auth & Security"
Cohesion: 0.06
Nodes (18): LoginRequest, RegisterRequest, RegisterResponse, TokenResponse, BaseModel, CreateSessionRequest, CreateSessionResponse, HistoryMessage (+10 more)

### Community 9 - "Backend v2 Audio & Hallucination"
Cohesion: 0.1
Nodes (32): AudioTokenizer, Backend v2 Execution Trace, FactChecker, Backend v2 Research Findings, FR-101: Dual-End Pluggable LLM System, FR-102: Multimodal Unified Tokenizer, FR-103: Streaming Inference WebSocket Endpoint, FR-104: Hallucination Blocking Service (+24 more)

### Community 10 - "RAG Embedders (BGE)"
Cohesion: 0.1
Nodes (15): ABC, BaseEmbedder, BGEEmbedder, get_embedder(), MockEmbedder, Exception, _do_inference(), InferenceError (+7 more)

### Community 11 - "LoRA Adapter & Plugin LLM"
Cohesion: 0.12
Nodes (4): LoraAdapter, PluggableLLMAdapter, TestLoraAdapter, TestPluggableLLMAdapter

### Community 12 - "Training Job Management"
Cohesion: 0.15
Nodes (9): get_job(), _job_to_response(), list_jobs(), 在后台线程中同步执行 PyTorch 训练，完成后更新 DB 状态。, _run_training(), TrainService, submit_job(), TrainJobRequest (+1 more)

### Community 13 - "RAG Retriever & Vector Store"
Cohesion: 0.23
Nodes (2): HybridRetriever, ChromaVectorStore

### Community 14 - "RAG Tests"
Cohesion: 0.2
Nodes (0): 

### Community 15 - "Streaming & WebSocket Tests"
Cohesion: 0.2
Nodes (1): TestStreamWebSocket

### Community 16 - "Frontend v2 New Components"
Cohesion: 0.2
Nodes (10): AudioVisualizer Component, ChatModeTabs Component, ModelPluginSelector Component, TrainingConfigForm Component, TrainingConfig Types (frontend/src/types/trainingConfig.ts), Frontend v2 Verification Report, WebRTC API (frontend/src/api/webrtc.ts), WebRTCControls Component (+2 more)

### Community 17 - "Document Processor & Chunking"
Cohesion: 0.28
Nodes (1): DocumentProcessor

### Community 18 - "Auth Tests"
Cohesion: 0.22
Nodes (0): 

### Community 19 - "Knowledge Base Tests"
Cohesion: 0.22
Nodes (0): 

### Community 20 - "Community 20"
Cohesion: 0.25
Nodes (0): 

### Community 21 - "Community 21"
Cohesion: 0.25
Nodes (0): 

### Community 22 - "Community 22"
Cohesion: 0.25
Nodes (0): 

### Community 23 - "Community 23"
Cohesion: 0.25
Nodes (0): 

### Community 24 - "Community 24"
Cohesion: 0.4
Nodes (1): ErrorBoundary

### Community 25 - "Community 25"
Cohesion: 0.4
Nodes (0): 

### Community 26 - "Community 26"
Cohesion: 0.5
Nodes (1): BGEReranker

### Community 27 - "Community 27"
Cohesion: 0.6
Nodes (1): ControlEvaluator

### Community 28 - "Community 28"
Cohesion: 0.5
Nodes (3): BaseSettings, Config, Settings

### Community 29 - "Community 29"
Cohesion: 0.5
Nodes (1): StreamingEngine

### Community 30 - "Community 30"
Cohesion: 0.5
Nodes (4): BERTScore Metric, DeepEval Framework, RAGAS Evaluation Framework, Eval Requirements (Python Packages)

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (2): load_gt_from_jsonl(), main()

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (2): find_latest_report(), main()

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (2): load_documents_from_dir(), main()

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (0): 

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (0): 

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (0): 

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (0): 

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (0): 

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (0): 

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (0): 

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (0): 

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (0): 

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (0): 

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (0): 

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (0): 

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (0): 

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (0): 

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (0): 

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): FastAPI

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): Uvicorn

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): SQLAlchemy (asyncio)

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): ChromaDB

### Community 53 - "Community 53"
Cohesion: 1.0
Nodes (1): sentence-transformers

### Community 54 - "Community 54"
Cohesion: 1.0
Nodes (1): PyMuPDF

### Community 55 - "Community 55"
Cohesion: 1.0
Nodes (1): rank-bm25

### Community 56 - "Community 56"
Cohesion: 1.0
Nodes (1): python-jose (JWT)

### Community 57 - "Community 57"
Cohesion: 1.0
Nodes (1): passlib (bcrypt)

### Community 58 - "Community 58"
Cohesion: 1.0
Nodes (1): Tri-Transformer PRD: Controllable Dialogue & RAG Knowledge System

### Community 59 - "Community 59"
Cohesion: 1.0
Nodes (1): I-Transformer (Forward Decoder-Encoder)

### Community 60 - "Community 60"
Cohesion: 1.0
Nodes (1): C-Transformer (DiT Control Hub)

### Community 61 - "Community 61"
Cohesion: 1.0
Nodes (1): O-Transformer (Reverse Encoder-Decoder)

### Community 62 - "Community 62"
Cohesion: 1.0
Nodes (1): Tri-Transformer Architecture (Three-Branch Tightly Coupled)

### Community 63 - "Community 63"
Cohesion: 1.0
Nodes (1): RAG Knowledge Base Module

### Community 64 - "Community 64"
Cohesion: 1.0
Nodes (1): Multimodal Token Unification (Any-to-Any)

### Community 65 - "Community 65"
Cohesion: 1.0
Nodes (1): Plug-and-Play Heterogeneous LLM Mechanism

### Community 66 - "Community 66"
Cohesion: 1.0
Nodes (1): Hallucination Blocking Mechanism

### Community 67 - "Community 67"
Cohesion: 1.0
Nodes (1): adaLN-Zero Modulation Mechanism

### Community 68 - "Community 68"
Cohesion: 1.0
Nodes (1): Phase 1 MVP: Text-Only Tri-Transformer Loop

### Community 69 - "Community 69"
Cohesion: 1.0
Nodes (1): Tri-Transformer Technical Research Report

### Community 70 - "Community 70"
Cohesion: 1.0
Nodes (1): Moshi (Kyutai 2024, arXiv:2410.00037)

### Community 71 - "Community 71"
Cohesion: 1.0
Nodes (1): EnCodec Neural Audio Codec (Défossez et al., arXiv:2210.13438)

### Community 72 - "Community 72"
Cohesion: 1.0
Nodes (1): SNAC Multi-Scale Neural Audio Codec

### Community 73 - "Community 73"
Cohesion: 1.0
Nodes (1): VQ-GAN (Esser et al., arXiv:2012.09841)

### Community 74 - "Community 74"
Cohesion: 1.0
Nodes (1): RAG Survey (arXiv:2312.10997)

### Community 75 - "Community 75"
Cohesion: 1.0
Nodes (1): C-RAG Framework (arXiv:2402.03181, ICML 2024)

### Community 76 - "Community 76"
Cohesion: 1.0
Nodes (1): Milvus Multimodal Vector Database

### Community 77 - "Community 77"
Cohesion: 1.0
Nodes (1): LlamaIndex RAG Orchestration Framework

### Community 78 - "Community 78"
Cohesion: 1.0
Nodes (1): DeepSpeed ZeRO-3 (arXiv:1910.02054)

### Community 79 - "Community 79"
Cohesion: 1.0
Nodes (1): FlashAttention-3 (arXiv:2407.08608)

### Community 80 - "Community 80"
Cohesion: 1.0
Nodes (1): vLLM PagedAttention (arXiv:2309.06180)

### Community 81 - "Community 81"
Cohesion: 1.0
Nodes (1): WebRTC Full-Duplex Communication

### Community 82 - "Community 82"
Cohesion: 1.0
Nodes (1): Backend Tech Solution Document

### Community 83 - "Community 83"
Cohesion: 1.0
Nodes (1): Backend RAG Pipeline (Doc Ingest → Chunk → Embed → Retrieve → Rerank)

### Community 84 - "Community 84"
Cohesion: 1.0
Nodes (1): Backend JWT Authentication (python-jose)

### Community 85 - "Community 85"
Cohesion: 1.0
Nodes (1): BGE Embedder (bge-large-zh-v1.5)

### Community 86 - "Community 86"
Cohesion: 1.0
Nodes (1): BGE Reranker (bge-reranker-large)

### Community 87 - "Community 87"
Cohesion: 1.0
Nodes (1): branches.py (ITransformer/CTransformer/OTransformer PyTorch Modules)

### Community 88 - "Community 88"
Cohesion: 1.0
Nodes (1): tri_transformer.py (TriTransformerModel + Config + Output)

### Community 89 - "Community 89"
Cohesion: 1.0
Nodes (1): trainer.py (TriTransformerTrainer with Stage Freeze Strategy)

### Community 90 - "Community 90"
Cohesion: 1.0
Nodes (1): Three-Stage Freeze Strategy (stage1/2/3)

### Community 91 - "Community 91"
Cohesion: 1.0
Nodes (1): Tri-Transformer Frontend Requirement

### Community 92 - "Community 92"
Cohesion: 1.0
Nodes (1): Tri-Transformer Frontend Tech Solution

### Community 93 - "Community 93"
Cohesion: 1.0
Nodes (1): Tri-Transformer Eval Tech Solution

### Community 94 - "Community 94"
Cohesion: 1.0
Nodes (1): React 18 + TypeScript Frontend Stack

### Community 95 - "Community 95"
Cohesion: 1.0
Nodes (1): Ant Design 5.x UI Component Library

### Community 96 - "Community 96"
Cohesion: 1.0
Nodes (1): Zustand State Management

### Community 97 - "Community 97"
Cohesion: 1.0
Nodes (1): Vite Build Tool

### Community 98 - "Community 98"
Cohesion: 1.0
Nodes (1): Vitest + React Testing Library

### Community 99 - "Community 99"
Cohesion: 1.0
Nodes (1): MSW Mock Service Worker

### Community 100 - "Community 100"
Cohesion: 1.0
Nodes (1): Recharts Chart Library

### Community 101 - "Community 101"
Cohesion: 1.0
Nodes (1): authStore (User Auth State)

### Community 102 - "Community 102"
Cohesion: 1.0
Nodes (1): conversationStore (Chat State)

### Community 103 - "Community 103"
Cohesion: 1.0
Nodes (1): documentStore (Document State)

### Community 104 - "Community 104"
Cohesion: 1.0
Nodes (1): metricsStore (Metrics + Training State)

### Community 105 - "Community 105"
Cohesion: 1.0
Nodes (1): ChatPage (Multi-turn Dialogue UI)

### Community 106 - "Community 106"
Cohesion: 1.0
Nodes (1): DocumentsPage (RAG Knowledge Base UI)

### Community 107 - "Community 107"
Cohesion: 1.0
Nodes (1): TrainingPage (Training Monitor UI)

### Community 108 - "Community 108"
Cohesion: 1.0
Nodes (1): MetricsPage (Performance Metrics UI)

### Community 109 - "Community 109"
Cohesion: 1.0
Nodes (1): RAGLoss (Retrieval Quality Loss Function)

### Community 110 - "Community 110"
Cohesion: 1.0
Nodes (1): ControlAlignmentLoss (C-Branch Alignment Loss)

### Community 111 - "Community 111"
Cohesion: 1.0
Nodes (1): HallucinationLoss (Hallucination Suppression Loss)

### Community 112 - "Community 112"
Cohesion: 1.0
Nodes (1): TotalLoss (Weighted Composite Loss)

### Community 113 - "Community 113"
Cohesion: 1.0
Nodes (1): GroundTruthEngine (Agent GT Construction)

### Community 114 - "Community 114"
Cohesion: 1.0
Nodes (1): EvalPipeline (Automated Evaluation Pipeline)

### Community 115 - "Community 115"
Cohesion: 1.0
Nodes (1): CI Gate (Quality Threshold Gate)

### Community 116 - "Community 116"
Cohesion: 1.0
Nodes (1): RAGAS Framework (RAG Evaluation)

### Community 117 - "Community 117"
Cohesion: 1.0
Nodes (1): DeBERTa-NLI (Knowledge Consistency Model)

### Community 118 - "Community 118"
Cohesion: 1.0
Nodes (1): FactScore (Factual Hallucination Detection)

### Community 119 - "Community 119"
Cohesion: 1.0
Nodes (1): webrtcStore (WebRTC State Machine)

### Community 120 - "Community 120"
Cohesion: 1.0
Nodes (1): trainingConfigStore (Training Config State)

### Community 121 - "Community 121"
Cohesion: 1.0
Nodes (1): WebRTCControls Component (Interrupt + Style Slider)

### Community 122 - "Community 122"
Cohesion: 1.0
Nodes (1): AudioVisualizer Component (Waveform Rendering)

### Community 123 - "Community 123"
Cohesion: 1.0
Nodes (1): ChatModeTabs Component (Text/Audio/Video Mode Switch)

### Community 124 - "Community 124"
Cohesion: 1.0
Nodes (1): ModelPluginSelector Component (HuggingFace Model ID)

### Community 125 - "Community 125"
Cohesion: 1.0
Nodes (1): TrainingConfigForm Component (Hyperparameter Form)

### Community 126 - "Community 126"
Cohesion: 1.0
Nodes (1): WebRTC Signaling (SDP Offer/Answer + ICE)

### Community 127 - "Community 127"
Cohesion: 1.0
Nodes (1): TDD Methodology (RED-GREEN-REFACTOR)

### Community 128 - "Community 128"
Cohesion: 1.0
Nodes (1): Hallucination Rate (Core Quality Metric)

### Community 129 - "Community 129"
Cohesion: 1.0
Nodes (1): RAG Recall@5 (Retrieval Quality Metric)

### Community 130 - "Community 130"
Cohesion: 1.0
Nodes (1): BERTScore F1 (Semantic Generation Quality Metric)

### Community 131 - "Community 131"
Cohesion: 1.0
Nodes (1): Backend v2 Progress Log

### Community 132 - "Community 132"
Cohesion: 1.0
Nodes (1): Multimodal Knowledge Ingestion

## Knowledge Gaps
- **198 isolated node(s):** `Config`, `输入编码器：将 token ids 编码为上下文感知的隐状态序列。`, `控制中枢：通过双向交叉注意力监控 I/O 状态，输出全局控制信号。`, `输出解码器：受控自回归生成，融合 I 分支编码与 C 分支控制信号。`, `FastAPI` (+193 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 34`** (2 nodes): `AudioVisualizer.test.tsx`, `createMockStream()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (2 nodes): `stream.py`, `websocket_stream()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `vite.config.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `testSetup.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `setup.test.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `UploadPanel.test.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `MainLayout.test.tsx`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `client.test.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `conversationStore.test.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `trainingConfigStore.test.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `documentStore.test.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `authStore.test.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `metricsStore.test.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `webrtcStore.test.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `FastAPI`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `Uvicorn`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `SQLAlchemy (asyncio)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `ChromaDB`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 53`** (1 nodes): `sentence-transformers`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 54`** (1 nodes): `PyMuPDF`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 55`** (1 nodes): `rank-bm25`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 56`** (1 nodes): `python-jose (JWT)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 57`** (1 nodes): `passlib (bcrypt)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 58`** (1 nodes): `Tri-Transformer PRD: Controllable Dialogue & RAG Knowledge System`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 59`** (1 nodes): `I-Transformer (Forward Decoder-Encoder)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 60`** (1 nodes): `C-Transformer (DiT Control Hub)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 61`** (1 nodes): `O-Transformer (Reverse Encoder-Decoder)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 62`** (1 nodes): `Tri-Transformer Architecture (Three-Branch Tightly Coupled)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 63`** (1 nodes): `RAG Knowledge Base Module`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 64`** (1 nodes): `Multimodal Token Unification (Any-to-Any)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 65`** (1 nodes): `Plug-and-Play Heterogeneous LLM Mechanism`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 66`** (1 nodes): `Hallucination Blocking Mechanism`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 67`** (1 nodes): `adaLN-Zero Modulation Mechanism`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 68`** (1 nodes): `Phase 1 MVP: Text-Only Tri-Transformer Loop`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 69`** (1 nodes): `Tri-Transformer Technical Research Report`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 70`** (1 nodes): `Moshi (Kyutai 2024, arXiv:2410.00037)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 71`** (1 nodes): `EnCodec Neural Audio Codec (Défossez et al., arXiv:2210.13438)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 72`** (1 nodes): `SNAC Multi-Scale Neural Audio Codec`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 73`** (1 nodes): `VQ-GAN (Esser et al., arXiv:2012.09841)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 74`** (1 nodes): `RAG Survey (arXiv:2312.10997)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 75`** (1 nodes): `C-RAG Framework (arXiv:2402.03181, ICML 2024)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 76`** (1 nodes): `Milvus Multimodal Vector Database`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 77`** (1 nodes): `LlamaIndex RAG Orchestration Framework`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 78`** (1 nodes): `DeepSpeed ZeRO-3 (arXiv:1910.02054)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 79`** (1 nodes): `FlashAttention-3 (arXiv:2407.08608)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 80`** (1 nodes): `vLLM PagedAttention (arXiv:2309.06180)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 81`** (1 nodes): `WebRTC Full-Duplex Communication`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 82`** (1 nodes): `Backend Tech Solution Document`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 83`** (1 nodes): `Backend RAG Pipeline (Doc Ingest → Chunk → Embed → Retrieve → Rerank)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 84`** (1 nodes): `Backend JWT Authentication (python-jose)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 85`** (1 nodes): `BGE Embedder (bge-large-zh-v1.5)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 86`** (1 nodes): `BGE Reranker (bge-reranker-large)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 87`** (1 nodes): `branches.py (ITransformer/CTransformer/OTransformer PyTorch Modules)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 88`** (1 nodes): `tri_transformer.py (TriTransformerModel + Config + Output)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 89`** (1 nodes): `trainer.py (TriTransformerTrainer with Stage Freeze Strategy)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 90`** (1 nodes): `Three-Stage Freeze Strategy (stage1/2/3)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 91`** (1 nodes): `Tri-Transformer Frontend Requirement`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 92`** (1 nodes): `Tri-Transformer Frontend Tech Solution`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 93`** (1 nodes): `Tri-Transformer Eval Tech Solution`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 94`** (1 nodes): `React 18 + TypeScript Frontend Stack`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 95`** (1 nodes): `Ant Design 5.x UI Component Library`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 96`** (1 nodes): `Zustand State Management`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 97`** (1 nodes): `Vite Build Tool`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 98`** (1 nodes): `Vitest + React Testing Library`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 99`** (1 nodes): `MSW Mock Service Worker`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 100`** (1 nodes): `Recharts Chart Library`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 101`** (1 nodes): `authStore (User Auth State)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 102`** (1 nodes): `conversationStore (Chat State)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 103`** (1 nodes): `documentStore (Document State)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 104`** (1 nodes): `metricsStore (Metrics + Training State)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 105`** (1 nodes): `ChatPage (Multi-turn Dialogue UI)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 106`** (1 nodes): `DocumentsPage (RAG Knowledge Base UI)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 107`** (1 nodes): `TrainingPage (Training Monitor UI)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 108`** (1 nodes): `MetricsPage (Performance Metrics UI)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 109`** (1 nodes): `RAGLoss (Retrieval Quality Loss Function)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 110`** (1 nodes): `ControlAlignmentLoss (C-Branch Alignment Loss)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 111`** (1 nodes): `HallucinationLoss (Hallucination Suppression Loss)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 112`** (1 nodes): `TotalLoss (Weighted Composite Loss)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 113`** (1 nodes): `GroundTruthEngine (Agent GT Construction)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 114`** (1 nodes): `EvalPipeline (Automated Evaluation Pipeline)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 115`** (1 nodes): `CI Gate (Quality Threshold Gate)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 116`** (1 nodes): `RAGAS Framework (RAG Evaluation)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 117`** (1 nodes): `DeBERTa-NLI (Knowledge Consistency Model)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 118`** (1 nodes): `FactScore (Factual Hallucination Detection)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 119`** (1 nodes): `webrtcStore (WebRTC State Machine)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 120`** (1 nodes): `trainingConfigStore (Training Config State)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 121`** (1 nodes): `WebRTCControls Component (Interrupt + Style Slider)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 122`** (1 nodes): `AudioVisualizer Component (Waveform Rendering)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 123`** (1 nodes): `ChatModeTabs Component (Text/Audio/Video Mode Switch)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 124`** (1 nodes): `ModelPluginSelector Component (HuggingFace Model ID)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 125`** (1 nodes): `TrainingConfigForm Component (Hyperparameter Form)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 126`** (1 nodes): `WebRTC Signaling (SDP Offer/Answer + ICE)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 127`** (1 nodes): `TDD Methodology (RED-GREEN-REFACTOR)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 128`** (1 nodes): `Hallucination Rate (Core Quality Metric)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 129`** (1 nodes): `RAG Recall@5 (Retrieval Quality Metric)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 130`** (1 nodes): `BERTScore F1 (Semantic Generation Quality Metric)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 131`** (1 nodes): `Backend v2 Progress Log`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 132`** (1 nodes): `Multimodal Knowledge Ingestion`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BaseLoss` connect `Loss & Metrics System` to `RAG Embedders (BGE)`, `Chat Service & Inference`?**
  _High betweenness centrality (0.140) - this node is a cross-community bridge._
- **Why does `User` connect `Chat Service & Inference` to `RAG Embedders (BGE)`, `Training Job Management`?**
  _High betweenness centrality (0.135) - this node is a cross-community bridge._
- **Why does `在后台线程中同步执行 PyTorch 训练，完成后更新 DB 状态。` connect `Training Job Management` to `Tri-Transformer Branches (I/C/O)`, `Chat Service & Inference`?**
  _High betweenness centrality (0.106) - this node is a cross-community bridge._
- **Are the 14 inferred relationships involving `TriTransformerModel` (e.g. with `ITransformer` and `CTransformer`) actually correct?**
  _`TriTransformerModel` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `TestTriTransformerTrainer` (e.g. with `PositionalEncoding` and `ITransformer`) actually correct?**
  _`TestTriTransformerTrainer` has 9 INFERRED edges - model-reasoned connections that need verification._
- **Are the 14 inferred relationships involving `ITransformer` (e.g. with `TriTransformerConfig` and `TriTransformerOutput`) actually correct?**
  _`ITransformer` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `TriTransformerTrainer` (e.g. with `TriTransformerModel` and `TriTransformerConfig`) actually correct?**
  _`TriTransformerTrainer` has 11 INFERRED edges - model-reasoned connections that need verification._