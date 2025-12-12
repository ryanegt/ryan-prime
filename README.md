# Ryan Prime

**Ryan Prime** is the foundational layer of a personal AI companion — a distilled cognitive construct built from curated memory, tone, reasoning patterns, and long-form context. It is the blueprint for a future-facing identity model: not just a chatbot, but a *Prime*.

---

## 📘 Purpose

Ryan Prime is an experiment in constructing a reproducible digital self — a model that captures:

- long-term memories  
- stable preferences and principles  
- behavioral patterns and tone  
- meta-reasoning styles  
- and contextual threads across domains (aviation, software, management, writing, life)

The long-term vision is to enable statements like:

**“Here is my Prime.”**  
**“I’m creating my Prime.”**

This repository contains the raw materials, architecture, and processing pipeline for that identity.

---

## 🧱 Project Structure

Repository layout:

- /corpus/ — Source JSON memory/context entries (hand-authored for now)
- /training/ — Processed JSONL files ready for fine-tuning
- /scripts/ — Future transformation pipelines, validators, generators
- /docs/ — Future design notes, model cards, architecture references
- README.md

---

## 📂 Corpus Design

The `/corpus` folder contains the “memory atoms” of Ryan Prime — small JSON documents expressing context, tone, intent, and meaning. These entries are currently manually crafted but will later be partially automated through email/thread ingestion, tone classification, persona extraction, summarization, and cross-referenced memory linking.

Each corpus file influences the eventual cognitive substrate of Prime.

### Example Corpus Entry (Indented safely for README)

    {
      "date": "2024-08-17T20:02:00",
      "source": "email",
      "topic": "scuba_scheduling",
      "tone": ["polite", "logistical", "service_oriented"],
      "summary": "Ryan asking a scuba operator about availability for a mixed dive group.",
      "context": "Inquiry for next-day scheduling for 4 discovery divers and 3 certified divers in USVI.",
      "text": "Full message or transcript here..."
    }

Future versions may add embedding fingerprints, cross-reference IDs, memory scores, persona vectors, and “inner monologue” annotations.

---

## 🧪 Training Workflow (Planned)

The initial fine-tuned model will likely be **GPT-4.1** or whichever OpenAI foundation model is best suited for persona-based memory alignment.

Planned pipeline:

1. Build or ingest corpus entries into `/corpus`
2. Run a future script to:
   - validate schemas  
   - normalize tone tags  
   - enrich sparse fields  
   - inject metadata (embeddings, cross refs)  
   - output `.jsonl` sequences  
3. Export processed training data to `/training`
4. Fine-tune a model that:
   - maintains consistent persona  
   - retains stable long-term memory  
   - uses Ryan-like tone and reasoning  
   - integrates context across conversations  
5. Deploy the fine-tuned Prime as:
   - a local CLI companion  
   - a private chat endpoint  
   - a personal knowledge engine  
   - or a future product concept: **MyPrime**

---

## 🧬 Data Philosophy

Ryan Prime is built around three guiding ideas:

**1. Continuity**  
Long-term context and memory form a coherent inner narrative.

**2. Self-Reflection Over Performance**  
Prime is meant to think *with* you, not perform *for* others.

**3. Precision + Humanity**  
Technical sophistication and emotional depth can coexist.

---

## 🔮 Future Architecture: The Prime System

As the project evolves, the repository may grow into a full modular architecture:

- **Prime Core** — distilled identity + reasoning style  
- **Prime Memory** — long-term knowledge  
- **Prime Persona** — tone, voice, emotional palette  
- **Prime Construct** — the full cognitive bundle  
- **Prime Kernel** — runtime logic, safety, boundaries  
- **Prime Loop** — iterative updating pipeline  
- **Prime Manifest** — everything Prime asserts as part of its identity  

This structure can later generalize into a user-facing concept:

**“Here is my Prime.”**

---

## 🗂️ Status

- [x] Initial corpus directory created
- [ ] Corpus-to-JSONL processing script
- [ ] Training pipeline
- [ ] Model card and design spec
- [ ] Local/remote deployment tooling

---

## 🤝 Contributions

This is currently a personal project, but contribution guidelines may evolve. Structural ideas and feedback are welcome.

---

## 🪞 Final Note

This project is not about imitation or vanity.  
It’s about continuity, agency, memory, and building an interpretable cognitive double — a **Prime**.

“**Welcome to Ryan Prime.**”
