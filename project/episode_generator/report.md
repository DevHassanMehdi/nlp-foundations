# Episode Generator Project Report  
**Author:** Hassan Mehdi  
**Date:** March 1, 2026  

## Abstract
This project aimed to build a local, end-to-end NLP application that generates structured TV episode drafts from a user prompt. I implemented and compared multiple generation pipelines, starting from classical probabilistic language modeling and progressing to neural and locally hosted transformer-based methods. The core system includes a data preparation pipeline, trainable models, a structured generation framework, and a live web interface (FastAPI backend + modern frontend) for side-by-side output comparison.

I began with a word-level n-gram baseline and a character-level neural model. Early outputs were mostly incoherent and often contained non-words, repeated fragments, or disconnected sentence transitions. To improve results, I switched from char-level generation to word-level LSTM, added decoding constraints, introduced planner-driven structured generation, tuned training aggressively for available hardware, and later integrated a local transformer model with fallback support. I also fixed engineering issues such as model-checkpoint mismatches, tokenizer/template compatibility, and prompt leakage in generated text.

Despite these improvements, the final outputs remained below the quality bar for reliable “full episode” coherence. The generated content has better format and section structure than initial versions, but plot continuity and semantic consistency still degrade across longer sections. I conclude that the work was technically valuable and educational, but quality remained unsatisfactory for the original ambition due to time and resource constraints. I am submitting the project in this state because the deadline was reached, while clearly documenting what worked, what failed, and what I learned.

---

## 1. Project Goal and Scope

The goal of this project was to create a practical NLP system that demonstrates the progression from foundational language modeling to modern generation approaches. I wanted something interactive and demonstrable: input a prompt, generate a complete episode draft, and compare model behavior side by side.

The main objectives were:

1. Build a local pipeline from data preparation to inference.
2. Train at least two models from scratch on my own prepared corpus.
3. Add a stronger local transformer model for comparison.
4. Generate outputs in a fixed, interpretable episode format.
5. Provide a web demo for real-time testing and qualitative evaluation.
6. Document performance-improvement attempts and final limitations.

I intentionally avoided external paid APIs and focused on reproducible local workflows, including model training and inference in my own environment.

---

## 2. Initial Design and Architecture

I designed the project with clear module boundaries:

- **Data layer:** corpus extraction/cleaning and tokenization.
- **Model layer:** n-gram and neural generation models.
- **Structured generation layer:** planner + formatter to enforce episode sections.
- **Application layer:** FastAPI endpoint for inference.
- **Frontend layer:** modern UI showing multiple model outputs side by side.

The episode format was not left open-ended; I enforced sections such as:

- Title
- Episode items (genre, setting, characters, theme)
- Logline
- Plot outline
- Scene breakdown
- Script sample

This structure was added because unconstrained generation often produced unusable output. The structure improved readability and made model comparison easier.

---

## 3. Development Timeline and Step-by-Step Changes

## 3.1 Baseline Phase: N-gram + Char-RNN

I started with two basic generation approaches:

1. **Word-level n-gram model**  
   - Fast to train.  
   - Easy to inspect and debug.  
   - Useful as a transparent baseline.  
   - Major weakness: no long-range context, high repetition, brittle phrasing.

2. **Character-level RNN**  
   - Initially appealing because it can model arbitrary text.  
   - In practice, generated many malformed or gibberish strings.  
   - Produced text-shaped output but poor lexical quality and weak semantics.

At this stage, generated outputs were mostly not acceptable as “episode drafts.” The outputs looked like language, but they were not coherent narratives.

---

## 3.2 First Major Pivot: Char-RNN to Word-LSTM

To reduce gibberish and improve lexical quality, I replaced char-level generation with **word-level LSTM**.

### What changed:
- Added word vocabulary handling with `<unk>`, padding, and sequence batching.
- Built training/inference code around tokenized word sequences.
- Tuned decoding with:
  - temperature
  - top-k sampling
  - repetition penalty
  - no-repeat n-gram blocking

### Result:
- Word validity improved significantly (fewer fake words).
- Sentence fragments became more readable.
- Coherence improved slightly but was still unstable over long sections.
- Repetition loops still occurred in some generations.

This pivot was the first clear quality improvement.

---

## 3.3 Structure Enforcement: Planner + Formatter Layer

Even with better tokens, narrative coherence remained weak. I introduced a **two-step structured generation approach**:

1. **Episode planner** generates a schema and section-level prompts.
2. **Section generator** fills each section independently.
3. **Formatter** assembles all sections into a single episode template.

### Why I added this:
- Long free-form generation frequently drifts.
- Section-specific prompts are easier for weaker models.
- Outputs become comparable across models.

### Result:
- The output format became consistently usable.
- Model behavior became easier to diagnose section by section.
- Coherence improved locally, but global plot consistency was still limited.

This was a major engineering win even though it did not fully solve language quality.

---

## 3.4 Training and Optimization Passes

I then focused on training quality and hardware utilization.

### Training improvements:
- GPU-aware training path.
- Automatic mixed precision (AMP).
- Gradient accumulation for larger effective batch sizes.
- AdamW optimizer.
- Learning-rate scheduling.
- Longer training runs (including high-epoch experiments).
- Adjusted sequence length, embedding size, hidden size, and dropout.

### Practical issues encountered:
- **Checkpoint/model mismatch** when architecture changed.
- Needed to save model config with checkpoints and load model from saved config.
- Runtime errors from missing class definitions/import order.

### Result:
- Training stability improved.
- Loss reduced compared to earlier runs.
- Generated text remained only moderately coherent.
- Improvements were incremental, not transformative.

This phase consumed substantial effort and highlighted that optimization alone cannot compensate for model/data limits in long-form generation tasks.

---

## 3.5 Local Transformer Model Integration

Since custom models still struggled, I added a third pipeline with a local transformer model.

### Iteration path:
- Initial transformer integration attempt had compatibility issues.
- Tried weaker fallback options that were still low quality.
- Switched to stronger GPT-style instruct model family with fallback tiers.
- Added safer loading and compatibility handling.

### Additional fixes:
- Prompt leakage issue: the model sometimes repeated instruction text in output.
- Fixed by decoding only newly generated tokens and post-cleaning instruction echoes.
- Added tokenizer template compatibility handling (`Tensor` vs `BatchEncoding` behaviors).

### Result:
- Best qualitative output among the three pipelines.
- Still not reliably coherent for full multi-section episodes.
- Better sentence-level fluency than custom models.
- Plot logic can still drift or contradict earlier sections.
- This pipeline was the strongest performer in my final side-by-side comparison.

---

## 3.6 Live Demo and UX Work

I implemented a live local demo:

- **Backend:** FastAPI `/generate` endpoint.
- **Frontend:** modern interface with side-by-side model panels.
- Prompt, temperature, seed controls.
- Displays outputs from:
  - n-gram model
  - word-LSTM model
  - transformer model

This made the system useful for fast iterative testing and direct qualitative comparison. It also improved project communication value, since results are immediately visible and interactive.

---

## 4. Technical Evaluation

This project was primarily evaluated qualitatively through prompt-based generation review, with emphasis on:

1. **Format compliance**  
   - Did the output follow required episode structure?
2. **Fluency**  
   - Are sentences grammatical and readable?
3. **Local coherence**  
   - Does each section make sense on its own?
4. **Global coherence**  
   - Are characters/events consistent across sections?
5. **Prompt adherence**  
   - Does output reflect user prompt intent?

### Observed model behavior

**N-gram baseline**
- Strengths:
  - Fast.
  - Predictable structure support.
- Weaknesses:
  - Repetitive phrasing.
  - Weak semantics and context memory.
  - Frequent unnatural transitions.

**Word-LSTM**
- Strengths:
  - Better lexical quality than char-level.
  - Better fluency than n-gram in many cases.
- Weaknesses:
  - Still drifts in long outputs.
  - Can repeat motifs and lose plot direction.
  - Inconsistent narrative causality.

**Transformer LM (Qwen2.5 Instruct)**
- Strengths:
  - Best sentence-level fluency.
  - Better instruction following.
  - More coherent local sections.
- Weaknesses:
  - Still leaks structure/instruction artifacts occasionally.
  - Section-to-section continuity still unreliable.
  - Not fully dependable for complete episode scripts.

Overall, structure quality improved substantially; semantic coherence improved modestly but not enough.

---

## 5. Key Problems and How I Addressed Them

## 5.1 Gibberish and malformed words
- Problem: early generations contained many meaningless strings.
- Fix: moved from char-level modeling to word-level LSTM.
- Outcome: major lexical improvement.

## 5.2 Repetition and loops
- Problem: repeated phrases and loops degraded readability.
- Fix: repetition penalty, top-k sampling, no-repeat n-gram constraints.
- Outcome: reduced but not eliminated.

## 5.3 Long-form drift
- Problem: full free-form episodes quickly lose consistency.
- Fix: planner-driven section generation.
- Outcome: better local structure, limited global coherence gains.

## 5.4 Model mismatch and runtime instability
- Problem: checkpoint incompatibility after architecture changes.
- Fix: store and load model config from checkpoint.
- Outcome: stable inference across retraining iterations.

## 5.5 Prompt leakage in transformer outputs
- Problem: instruction text appeared in generated sections.
- Fix: decode only new tokens, strip instruction boilerplate, chat-template handling.
- Outcome: significantly reduced leakage.

---

## 6. What I Learned

This project produced strong practical learning outcomes, even with unsatisfactory final text quality.

### I learned:

1. **Architecture matters more than small hyperparameter tweaks.**  
   Moving from char-level to word-level modeling had a larger impact than many optimizer-level adjustments.

2. **Structure helps weaker generators.**  
   Planner + formatter design significantly improves output usability even when raw generation is weak.

3. **Training improvements have diminishing returns.**  
   Longer epochs and larger settings improved loss and fluency incrementally, but not enough for stable long-form narrative quality.

4. **Engineering reliability is critical in NLP projects.**  
   Checkpoint configs, tokenizer behavior, and generation decoding details can break the whole system if not handled carefully.

5. **Qualitative evaluation is hard but necessary for generative tasks.**  
   A model can look good on local sections but fail globally in story logic.

6. **Transformer models help, but integration quality matters.**  
   Prompt format, template handling, and output cleaning strongly influence final user-facing quality.

---

## 7. Limitations

The final system has clear limitations:

- The custom models are too weak for robust long-form episode generation.
- Training data quality/scale was not enough for complex narrative consistency.
- Evaluation was mostly qualitative; no large human-rated benchmark was run.
- Inference-time constraints improve cleanliness but can also reduce creativity.
- Local transformer models improve fluency but still fail at fully consistent multi-section storytelling in this setup.

---

## 8. Future Work

If I had more time, I would prioritize:

1. Curating a cleaner, domain-specific script dataset with better narrative signals.
2. Fine-tuning a stronger instruction model directly on episode/script-style data.
3. Adding reranking and consistency checks across sections.
4. Using retrieval-augmented context memory for characters/events.
5. Introducing automatic quality gates and regenerate-on-failure loops.
6. Adding formal human evaluation rubrics for coherence and creativity.

These steps are more likely to produce meaningful end-quality gains than simply extending current training epochs.

---

## 9. Conclusion

This project successfully delivered a complete local NLP application with multiple generation pipelines, structured output formatting, and a live side-by-side comparison interface. I implemented the full workflow from data preparation and model training to inference and UI integration, and I iteratively improved the system through several technical pivots.

I did not achieve satisfactory final episode quality. Although output structure became solid and fluency improved (especially with the transformer model), full narrative coherence remained inconsistent. Because the submission deadline was reached, I am leaving the project in this state.

Even so, the project was valuable: I explored classical and neural language generation, learned practical model engineering, handled real integration issues, and built a demonstrable system that clearly reflects my NLP development process and technical growth. The final result is best viewed as a strong learning artifact and an honest record of iterative experimentation under realistic time constraints.

---

## References

1. Jurafsky, D., & Martin, J. H. *Speech and Language Processing* (draft editions).  
2. PyTorch Documentation: training loops, mixed precision, optimization.  
3. Hugging Face Transformers Documentation: tokenizer/model loading and generation.  
4. FastAPI Documentation: API development and local serving.  
5. NLTK Documentation: corpus utilities and tokenization foundations.
