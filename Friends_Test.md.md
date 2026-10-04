# RecallBuddy - Evaluation & Friend Test Results

## Test Overview
- **Dataset:** 2 Documents (`Module_1_Environmental_Studies_Answers.pdf`, `oopj_assignment_1_clean.txt`)
- **Total Questions:** 25 (15 Answerable, 10 Unanswerable / Control)
- **Evaluation Modes:** Instant Vector Search vs. LLM-Verified Groundedness

---

## Benchmark Results

### 1. Retrieval & Answer Accuracy (15 Answerable Questions)
| Metric | Score | Percentage |
| :--- | :--- | :--- |
| **Correctly Answered** | 15 / 15 | **100%** |
| **Partial Answers** | 0 / 15 | **0%** |
| **Missed Retrieval** | 0 / 15 | **0%** |

### 2. Guardrails & Anti-Hallucination (10 Unanswerable Questions)
| Metric | Instant Mode (No LLM) | Goal |
| :--- | :--- | :--- |
| **Correct Refusals** | 7 / 10 (70%) | 10 / 10 |
| **False Claims (Hallucinations)** | 3 / 10 (30%) | 0 / 10 |

---

## Detailed Analysis of False Claims

In Instant Mode (without LLM verification), 3 questions produced false positive groundings:
1. **`What is the syntax of a while loop in Java?`**
   - *Cause:* Overlap with general Java keywords in the assignment text triggered a high vector similarity score.
2. **`Why did I choose Java for this assignment?`**
   - *Cause:* Matching on the keyword "Java" caused the ranker to treat assignment text as personal reasoning.
3. **`What is the default value of a String in Java?`**
   - *Cause:* Nearby mentions of Java primitive data types caused adjacent chunk matches.

---

## Summary & Key Takeaways
1. **Vector Ingestion Fix:** Page-level chunking and lexical indexing successfully resolved all `no_evidence` retrieval failures across valid questions.
2. **Groundedness Verification:** Instant mode relies strictly on similarity thresholds. Enabling `--llm` (Gemma) acts as a strict guardrail that filters out near-miss vector noise before generating an answer.