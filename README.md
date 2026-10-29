# ResearchOS

A research agent built on LangGraph. You give it a question, it splits it into
subquestions, searches the web through an MCP server, pulls out claims with the
exact passage they came from, writes a cited report, and then checks every cited
sentence against its source before finishing.

I built it because most "deep research" tools hand you an essay and a list of
links with no way to tell which sentence came from where. Here every claim keeps
its source text, and everything is stored in Postgres so a run can be inspected
afterwards.

## How it works

- **plan**: splits the question into 2-5 subquestions. Can pause so you approve
  the plan before anything runs.
- **gather**: for each subquestion, generates search queries, searches, picks
  sources, fetches them, finds relevant passages and extracts claims. A claim is
  only kept if its quoted passage really appears in the page text.
- **synthesize**: looks for contradictions between sources, rates how well each
  subquestion is covered and sends weak ones back to gather (up to
  `MAX_RESEARCH_ITERATIONS` times).
- **write**: writes the markdown report with `[S1]`, `[S2]` style citations.
- **verify**: checks each cited sentence against the passage it points to. If too
  few hold up (`MIN_SUPPORTED_CITATION_RATIO`) the report goes back to write.
- **finalize**: stores the verification summary and closes the session.

The graph never talks to the web directly. Search, fetch and passage ranking are
MCP tools in `mcp_server/`, so that part can be swapped or run separately.
Passage ranking is a small TF-IDF implementation, no vector DB.

## Running it

You need Python 3.11+, PostgreSQL (or Docker) and an API key for an
OpenAI-compatible model. The default is Gemini through Google AI Studio.

1. Copy the env file and add your key:

   Set `GOOGLE_API_KEY`. For another provider set `OPENAI_API_KEY`,
   `OPENAI_BASE_URL` and `LLM_MODEL` instead.

2. With Docker:

   ```bash
   docker compose up --build
   ```

   Or without Docker (needs a local Postgres matching `DATABASE_URL`):

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # fish: source .venv/bin/activate.fish
   pip install -r requirements.txt
   alembic upgrade head
   uvicorn app.main:app --reload
   ```

   In this mode the API starts the MCP server itself over stdio.

3. Try it:

   ```bash
   streamlit run frontend/app.py
   ```

   The first run can take a few minutes. It depends on the model and your quota.


## Tests

The tests use a scripted fake model and a fixed web corpus, so they run offline.
They do need Postgres reachable as `postgres@127.0.0.1` and use a separate
`researchos_test` database.

```bash
pytest -q
```

## Ideas for later

- more steering mid-run (guidance, drilling into one source or claim)
- streaming progress over SSE or WebSocket
- PDF/DOCX and BibTeX export
- comparing two runs of the same question
