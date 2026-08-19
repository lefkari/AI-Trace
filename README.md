# AI Trace V3 — Digital Authenticity Intelligence

## Current product
Evidence-first browser AI-text analysis.

### Stack
- GitHub Pages compatible static frontend
- Transformers.js 3.8.1
- ONNX Runtime Web
- `onnx-community/tmr-ai-text-detector-ONNX`
- No paid API / backend in V3
- Browser model cache

### V3 improvements
- International English-first UI
- Trace DNA score presentation
- Explicit confidence / uncertainty layer
- Segment-level Trace Map
- Evidence panel
- Document profile
- High-stakes-use warning
- Fallback mode if ML cannot load

### Important
AI detection is probabilistic. Never present the score as proof of authorship.

### Deploy
Upload/replace `index.html`, `style.css`, `app.js`, `README.md` in the root of the GitHub Pages repository and commit to `main`.

### Roadmap
1. Benchmark harness and calibration
2. Additional independent detector / ensemble
3. C2PA + metadata verification for images
4. Image forensic model
5. Audio/deepfake analysis
6. Video pipeline
7. Accounts, reports, API and monetization
