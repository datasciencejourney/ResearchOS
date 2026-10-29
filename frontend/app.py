"""Streamlit frontend for ResearchOS.

Run the API first, then:  streamlit run frontend/app.py
"""

import os
import time
import uuid

import httpx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

DEFAULT_API = os.environ.get("RESEARCHOS_API", "http://127.0.0.1:8000/api/v1")
COOKIE_NAME = "researchos_client"
DONE = {"completed", "failed"}
WAITING = {"awaiting_approval", "created"}

st.set_page_config(page_title="ResearchOS", page_icon="🔎", layout="wide")


# ---------- visitor id (kept in a cookie) ----------
def client_id() -> str:
    """Random id per browser. It is read from a cookie and sent to the API so
    each visitor only sees their own sessions."""
    if "client_id" in st.session_state:
        return st.session_state["client_id"]
    cid = None
    try:
        cid = st.context.cookies.get(COOKIE_NAME)
    except Exception:  # older Streamlit without st.context
        pass
    if not cid:
        cid = uuid.uuid4().hex
        # Streamlit can't set cookies itself, so a tiny script does it in the browser.
        components.html(
            "<script>try{window.parent.document.cookie="
            f"'{COOKIE_NAME}={cid}; max-age=31536000; path=/; SameSite=Lax';"
            "}catch(e){}</script>",
            height=0,
        )
    st.session_state["client_id"] = cid
    return cid


# ---------- api helpers ----------
def api(method: str, path: str, **kwargs):
    base = st.session_state.get("api_url", DEFAULT_API).rstrip("/")
    try:
        headers = {"X-Client-Id": client_id()}
        resp = httpx.request(method, base + path, headers=headers, timeout=60.0, **kwargs)
    except httpx.HTTPError as exc:
        st.error(f"Can't reach the API at {base}. Is it running? ({exc.__class__.__name__})")
        st.stop()
    if resp.status_code == 404:
        return None
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        st.error(f"API error {resp.status_code}: {detail}")
        st.stop()
    return resp.json()


def get(path: str, **params):
    return api("GET", path, params=params or None)


# ---------- sidebar ----------
with st.sidebar:
    st.title("ResearchOS")
    st.session_state.setdefault("api_url", DEFAULT_API)
    st.text_input("API address", key="api_url")

    if st.button("➕ New research", width="stretch"):
        st.session_state.pop("session_id", None)
        st.rerun()

    st.subheader("Past sessions")
    sessions = get("/sessions", limit=30) or []
    if not sessions:
        st.caption("Nothing yet.")
    for s in sessions:
        label = f"{s['question'][:42]}{'…' if len(s['question']) > 42 else ''}"
        icon = {"completed": "✅", "failed": "❌", "awaiting_approval": "⏸️"}.get(s["status"], "⏳")
        if st.button(f"{icon} {label}", key=f"pick-{s['id']}", width="stretch"):
            st.session_state["session_id"] = s["id"]
            st.rerun()


# ---------- new session form ----------
def new_session_form():
    st.header("Ask a research question")
    st.write(
        "ResearchOS splits the question into subquestions, searches the web, extracts claims "
        "with their source passages, writes a cited report and checks every citation."
    )
    with st.form("new"):
        question = st.text_area(
            "Question",
            placeholder="Is PostgreSQL a better choice than MongoDB for a SaaS app?",
            height=110,
        )
        approval = st.checkbox("Pause so I can approve the plan before research starts")
        submitted = st.form_submit_button("Start research", type="primary")
    if submitted:
        if len(question.strip()) < 8:
            st.warning("Please write a longer question (at least 8 characters).")
            return
        created = api(
            "POST",
            "/sessions",
            json={"question": question.strip(), "config": {"approval_required": approval}},
        )
        api("POST", f"/sessions/{created['id']}/run", json={"start": True})
        st.session_state["session_id"] = created["id"]
        st.rerun()


# ---------- session view ----------
def show_session(session_id: str):
    detail = get(f"/sessions/{session_id}")
    if detail is None:
        st.warning("That session no longer exists.")
        st.session_state.pop("session_id", None)
        return

    status = detail["status"]
    counts = detail.get("counts", {})
    st.header(detail["question"])

    cols = st.columns(5)
    cols[0].metric("Status", status.replace("_", " "))
    cols[1].metric("Sources", counts.get("sources", 0))
    cols[2].metric("Claims", counts.get("claims", 0))
    cols[3].metric("Contradictions", counts.get("contradictions", 0))
    cols[4].metric("Reports", counts.get("reports", 0))

    if status == "failed":
        st.error(detail.get("error") or "The run failed.")
        if st.button("Retry this run"):
            api("POST", f"/sessions/{session_id}/run", json={"start": True})
            st.rerun()
    elif status == "awaiting_approval":
        st.info("The plan is ready. Review it in the Plan tab, then approve to start research.")
        if st.button("Approve plan and continue", type="primary"):
            api("POST", f"/sessions/{session_id}/plan/approve")
            st.rerun()
    elif status not in DONE and status not in WAITING:
        st.progress(
            {"planning": 0.1, "researching": 0.45, "writing": 0.8, "verifying": 0.9}.get(status, 0.3),
            text=f"Working: {status}…",
        )

    tabs = st.tabs(["Report", "Plan", "Sources", "Claims", "Contradictions", "Citation check", "Trace"])

    with tabs[0]:
        report = get(f"/sessions/{session_id}/report")
        if report:
            summary = detail.get("verification_summary") or report.get("verification_summary") or {}
            if summary:
                ratio = summary.get("supported_ratio")
                passed = summary.get("passed")
                if ratio is not None:
                    st.caption(
                        f"Report v{report['version']} · citation check "
                        f"{'passed' if passed else 'did not pass'} · {ratio:.0%} of cited sentences supported"
                    )
            st.markdown(report["markdown"])
            st.download_button(
                "Download report (.md)",
                report["markdown"],
                file_name=f"report-{session_id[:8]}.md",
                mime="text/markdown",
            )
        else:
            st.caption("The report appears here when it has been written.")

    with tabs[1]:
        plan = get(f"/sessions/{session_id}/plan")
        if plan:
            st.caption(f"Plan v{plan['version']} · {plan['status']}")
            if plan.get("rationale"):
                st.write(plan["rationale"])
            for item in plan["items"]:
                suff = item.get("sufficiency")
                with st.expander(
                    f"{item['position'] + 1}. {item['subquestion']}  "
                    f"({item['status']}{'' if suff is None else f', evidence {suff:.2f}'})"
                ):
                    st.write(f"**Why:** {item['rationale']}")
                    st.write(f"**Evidence goal:** {item['evidence_goal']}")
        else:
            st.caption("No plan yet.")

    with tabs[2]:
        sources = get(f"/sessions/{session_id}/sources") or []
        if sources:
            for src in sources:
                excluded = src["is_excluded"]
                left, right = st.columns([6, 1])
                with left:
                    title = src["title"] or src["domain"]
                    st.markdown(f"**[{title}]({src['url']})**{' _(excluded)_' if excluded else ''}")
                    st.caption(
                        f"{src['domain']} · quality {src['quality_score']} · "
                        f"{src['quality_notes'] or 'no notes'}"
                    )
                with right:
                    label = "Include" if excluded else "Exclude"
                    if st.button(label, key=f"ex-{src['id']}"):
                        api(
                            "POST",
                            f"/sessions/{session_id}/sources/{src['id']}/exclude",
                            params={"excluded": not excluded},
                        )
                        st.rerun()
        else:
            st.caption("No sources yet.")

    with tabs[3]:
        claims = get(f"/sessions/{session_id}/claims") or []
        if claims:
            for claim in claims:
                with st.expander(f"[{claim['claim_type']}] {claim['text'][:110]}"):
                    st.write(claim["text"])
                    for ev in claim.get("evidence", []):
                        passage = ev.get("passage") or ev.get("text") or ""
                        if passage:
                            st.markdown(f"> {passage}")
        else:
            st.caption("No claims yet.")

    with tabs[4]:
        contradictions = get(f"/sessions/{session_id}/contradictions") or []
        if contradictions:
            for c in contradictions:
                st.markdown(f"**{c['topic']}** · severity: {c['severity']}")
                a, b = st.columns(2)
                a.info(f"{c['source_a_domain']}\n\n{c['claim_a_text']}")
                b.warning(f"{c['source_b_domain']}\n\n{c['claim_b_text']}")
                if c.get("context"):
                    st.caption(c["context"])
        else:
            st.caption("No contradictions found.")

    with tabs[5]:
        report = get(f"/sessions/{session_id}/report")
        if report:
            rows = get(f"/sessions/{session_id}/report/{report['version']}/verifications") or []
            if rows:
                ok = sum(r["verdict"] == "supported" for r in rows)
                st.write(f"**{ok} of {len(rows)}** sentences supported by their cited source.")
                st.dataframe(
                    pd.DataFrame(
                        [{"verdict": r["verdict"], "sentence": r["claim_text"], "notes": r["notes"]} for r in rows]
                    ),
                    width="stretch",
                    hide_index=True,
                )
        else:
            st.caption("Available once a report exists.")

    with tabs[6]:
        events = get(f"/sessions/{session_id}/events") or []
        if events:
            st.dataframe(
                pd.DataFrame(
                    [{"time": e["created_at"], "node": e["node"], "kind": e["kind"], "message": e["message"]} for e in events]
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("No events yet.")

    # keep refreshing while the run is active
    if status not in DONE and status not in WAITING:
        time.sleep(3)
        st.rerun()


# ---------- main ----------
if st.session_state.get("session_id"):
    show_session(st.session_state["session_id"])
else:
    new_session_form()