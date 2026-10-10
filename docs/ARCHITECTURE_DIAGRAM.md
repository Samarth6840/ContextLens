# ContextLens — Architecture Diagram

Rendered by GitHub / any Mermaid viewer.

## Full system

```mermaid
flowchart TB
    IN["Video + Audio Input"]

    subgraph L1["LAYER 1 — Multimodal Perception (src/layer1)"]
        direction TB
        SCENE["Scene object detection<br/>YOLO-World zero-shot<br/>detector.py"]
        LOGO["Logo detection<br/>YOLO-World brand queries<br/>logo_detector.py"]
        OCR["OCR<br/>PaddleOCR<br/>ocr.py / ocr_worker.py"]
        STT["Speech-to-text<br/>faster-whisper + brand mentions<br/>audio.py"]
        EVT["Audio events / quality<br/>librosa + BEATs<br/>audio.py"]
        RETR["Logo retrieval index<br/>CLIP crop→brand<br/>logo_retrieval.py"]
        OV["Open-vocab merge<br/>openset.py"]
    end

    subgraph L2["LAYER 2 — Intelligence (src/layer2)"]
        direction TB
        L2A["2a Quality-aware fusion<br/>dynamic cross-modal weights<br/>fusion.py"]
        L2B["2b Evidence confidence<br/>decomposed evidence_breakdown<br/>confidence.py"]
        L2C["2c Temporal memory / brand timeline<br/>cross-scene + cross-video<br/>brand_resolver.py, brand_memory.py"]
        L2D["2d Creator profiling (Phase 2)<br/>affinity + niche suppression<br/>creator_profiling.py"]
    end

    subgraph L3["LAYER 3 — Recommendation (src/layer3)"]
        KG["KnowledgeGraph<br/>brand→category→adjacent<br/>knowledge_graph.py"]
        REC["BrandRecommender<br/>DIRECT + SUGGESTED + reasons<br/>recommender.py"]
        AFF["Creator affinity model<br/>affinity.py, affinity_trainer.py"]
    end

    SRV["server.py / app.py<br/>dashboard, /api/analyse, outreach"]
    OUT["Dashboard JSON + outreach drafts"]

    IN --> L1
    SCENE --> OV
    LOGO --> RETR
    RETR --> L2C
    OV --> L2A
    OCR --> L2A
    STT --> L2A
    EVT --> L2A
    L2A --> L2B
    L2C --> L2B
    L2B --> L3
    L2C --> L3
    AFF --> REC
    KG --> REC
    L2D --> L3
    L3 --> SRV --> OUT
```

## Phase 1 request flow (implemented end-to-end)

```mermaid
sequenceDiagram
    participant U as User
    participant S as server.py
    participant P as Phase1Pipeline
    participant L1 as Layer 1
    participant L2 as Layer 2 a/b/c
    participant L3 as Layer 3

    U->>S: POST /api/analyse (upload)
    S->>S: queue job (background thread)
    S->>P: process_video(path, frame_rate=1.0)
    P->>L1: frames@1fps → scene det + logo det + OCR; audio → STT + events
    L1-->>P: aligned signals
    P->>P: BrandResolver: class-match → crop-OCR → CLIP retrieval → unresolved
    P->>L2: build_brand_timeline + weighted fusion
    L2-->>P: evidence_breakdown, confidence
    P->>L3: recommend(detected brands)
    L3-->>P: ranked DIRECT + SUGGESTED recs (with reasons)
    P-->>S: dashboard dict
    U->>S: GET /api/pipeline/<job>
    S-->>U: dashboard JSON
```

Source of truth for the annotated deliverable mapping: `docs/ARCHITECTURE_AND_RESEARCH.md`.
