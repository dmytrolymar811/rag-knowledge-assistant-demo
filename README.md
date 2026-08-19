# Document-Based RAG Knowledge Assistant

A small, production-minded demonstration of the retrieval layer behind a document-grounded AI assistant. It indexes local text documents, ranks relevant passages with TF-IDF cosine similarity, and returns traceable source references.

This repository deliberately focuses on retrieval and grounding. It does **not** claim to retrain a language model, and it does not require paid API access.

## What it demonstrates

- document loading and chunking;
- deterministic local indexing;
- relevant-passage retrieval;
- source citations and confidence scores;
- safe fallback when the knowledge base has no supported answer;
- testable Python architecture.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
python -m rag_assistant.cli "How can a customer reschedule an appointment?"
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

## Example

```text
Question: How can a customer reschedule an appointment?

Grounded context:
Customers can reschedule through the booking link in their confirmation email...

Sources:
- booking_policy.txt (score: 0.53)
```

## Architecture

```text
documents -> validated loader -> chunks -> TF-IDF index -> ranked passages
                                                       -> citations
                                                       -> safe fallback
```

The retrieval component can later be connected to GPT, Gemini, Claude, or a local model. API credentials, hosting, access control, and production data handling should be configured separately for each deployment.

## Limitations

- TXT and Markdown are supported in this compact demo.
- Retrieval quality depends on document quality and query wording.
- No answer-generation model is bundled, so the CLI returns grounded context rather than inventing a synthesized answer.
- This is a portfolio demonstration, not a hosted production service.

## License

MIT
