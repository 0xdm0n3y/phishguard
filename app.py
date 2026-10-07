
import csv
import datetime
import os

import joblib
import pandas as pd
import streamlit as st

from phish_utils import ADVICE, analyze, highlight_html, parse_eml
from demo_emails import DEMOS


MODEL_PATH = "model.joblib"
FEEDBACK_PATH = "feedback.csv"


st.set_page_config(
    page_title="PhishGuard",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("PhishGuard")
st.caption("Analyze an email for signs of phishing before you click, reply, or share information.")

if not os.path.exists(MODEL_PATH):
    st.error(
        "The trained model could not be found. "
        "Run `python train_model.py --data combined_phishing_dataset.csv` first."
    )
    st.stop()

art = load_model()

tab_paste, tab_eml = st.tabs(["Paste email", "Upload .eml"])

email = None

with tab_paste:
    c1, c2 = st.columns(2)

    sender = c1.text_input(
        "From",
        placeholder="support@example.com",
        key="f_sender",
    )

    reply_to = c2.text_input(
        "Reply-To",
        placeholder="reply@example.com",
        key="f_reply",
    )

    subject = st.text_input(
        "Subject",
        placeholder="Your account has been suspended",
        key="f_subject",
    )

    body = st.text_area(
        "Email body",
        height=240,
        key="f_body",
        placeholder="Paste the email contents here, including any links.",
    )

    if st.button("Analyze email", type="primary", key="go_paste"):
        if not (subject.strip() or body.strip()):
            st.warning("Add a subject or email body first.")
        else:
            email = {
                "subject": subject,
                "sender": sender,
                "reply_to": reply_to,
                "auth": "",
                "body": body,
                "links": [],
            }


with tab_eml:
    st.caption("Upload the original .eml message to include available headers and metadata.")

    up = st.file_uploader(
        "Choose an email file",
        type=["eml"],
    )

    if up is not None and st.button("Analyze file", type="primary", key="go_eml"):
        try:
            email = parse_eml(up.getvalue())
        except Exception as e:
            st.error(f"Could not read the email: {e}")


if email is not None:
    st.session_state["email"] = email
    st.session_state["result"] = analyze(art, email)
    st.session_state.pop("fb_done", None)


res = st.session_state.get("result")

if res:
    st.divider()

    lvl = res["level"]
    pct = res["final"] * 100

    color = {
        "LOW": "#2e7d32",
        "MEDIUM": "#ed8b00",
        "HIGH": "#c62828",
    }[lvl]

    verdict = {
        "LOW": "No obvious signs of phishing",
        "MEDIUM": "Some suspicious signs detected",
        "HIGH": "Strong signs of phishing",
    }[lvl]

    a, b = st.columns([1, 2])

    with a:
        st.markdown(
            f"""
            <div style="
                background:{color};
                color:white;
                padding:24px;
                border-radius:14px;
                text-align:center;
                margin-bottom:10px;
            ">
                <div style="font-size:14px;opacity:.9;letter-spacing:1px;">
                    {lvl} RISK
                </div>
                <div style="font-size:48px;font-weight:700;line-height:1.1;">
                    {pct:.0f}%
                </div>
                <div style="font-size:17px;margin-top:5px;">
                    {verdict}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with b:
        st.subheader("What you should do")

        for line in ADVICE[lvl]:
            st.markdown(f"- {line}")

    left, right = st.columns(2)

    with left:
        st.subheader("What we found")

        if res["flags"]:
            for sev, msg in res["flags"]:
                icon = "🔴" if sev == "high" else "🟠"
                st.markdown(f"{icon} {msg}")
        else:
            st.success("No rule-based warning signs were found.")

        if res["positives"]:
            for msg in res["positives"]:
                st.markdown(f"✓ {msg}")

        if res["trusted"]:
            st.info(
                f"The sender appears to be authenticated, so the text score was "
                f"reduced from {res['ml_raw'] * 100:.0f}% to "
                f"{res['ml_prob'] * 100:.0f}%."
            )

        if res.get("quiet"):
            st.info(
                f"No red flags, no links and no pressure language, so there is "
                f"little additional evidence supporting the model's prediction. "
                f"Its score was reduced from {res['ml_raw']*100:.0f}% to "
                f"{res['ml_prob']*100:.0f}% because short, ordinary messages can "
                f"be difficult for a text-based model to classify reliably."
            )

    with right:
        st.subheader("Why this score?")

        if res["phish_terms"]:
            st.markdown("**Signals associated with phishing**")

            st.dataframe(
                pd.DataFrame(
                    res["phish_terms"],
                    columns=["Term", "Weight"],
                ),
                hide_index=True,
                width="stretch",
            )

        if res["legit_terms"]:
            st.markdown("**Signals associated with legitimate email**")

            st.dataframe(
                pd.DataFrame(
                    res["legit_terms"],
                    columns=["Term", "Weight"],
                ),
                hide_index=True,
                width="stretch",
            )

    st.subheader("Detection scores")

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric("Final", f"{res['final'] * 100:.1f}%")
    m2.metric("ML", f"{res['ml_prob'] * 100:.1f}%")
    m3.metric("Random Forest", f"{res['p_rf'] * 100:.1f}%")
    m4.metric("Logistic Regression", f"{res['p_lr'] * 100:.1f}%")
    m5.metric("Rules", f"{res['rule_score'] * 100:.1f}%")
    st.caption(
    "Final score combines the machine-learning prediction with "
    "rule-based checks."
    )

    st.subheader("Email analysis")

    st.markdown(
        highlight_html(
            res["text"],
            [t for t, _ in res["phish_terms"]],
        ),
        unsafe_allow_html=True,
    )

  
    st.divider()

    st.subheader("Help improve the detector")
    st.caption(
        "If you know the correct classification, you can submit it for future retraining."
    )

    f1, f2 = st.columns(2)

    label = None

    if f1.button("This was phishing", use_container_width=True):
        label = 1

    if f2.button("This was legitimate", use_container_width=True):
        label = 0

    if label is not None:
        new = not os.path.exists(FEEDBACK_PATH)

        with open(
            FEEDBACK_PATH,
            "a",
            newline="",
            encoding="utf-8",
        ) as fh:
            w = csv.writer(fh)

            if new:
                w.writerow(
                    [
                        "timestamp",
                        "text",
                        "model_score",
                        "true_label",
                    ]
                )

            w.writerow(
                [
                    datetime.datetime.now().isoformat(timespec="seconds"),
                    res["text"],
                    f"{res['final']:.3f}",
                    label,
                ]
            )

        st.session_state["fb_done"] = True

    if st.session_state.get("fb_done"):
        st.success("Feedback saved.")


    report = (
        f"PhishGuard report\n"
        f"Risk: {lvl} ({pct:.0f}%)\n\n"
        f"Red flags:\n"
        + "\n".join(
            f"- [{sv}] {m}"
            for sv, m in res["flags"]
        )
        + "\n\n"
        + "Signals associated with phishing: "
        + ", ".join(t for t, _ in res["phish_terms"])
    )

    st.download_button(
        "Download report",
        report,
        file_name="phishguard_report.txt",
    )


def _load_demo():
    d = DEMOS[st.session_state["demo_pick"]]

    st.session_state.update(
        f_sender=d["sender"],
        f_reply=d["reply_to"],
        f_subject=d["subject"],
        f_body=d["body"],
    )


with st.sidebar:
    st.markdown("### Demo")

    st.selectbox(
        "Sample email",
        list(DEMOS),
        key="demo_pick",
    )

    st.button(
        "Load sample",
        on_click=_load_demo,
        use_container_width=True,
    )

    st.divider()

    st.markdown("### About")

    st.write(
        "PhishGuard combines machine-learning predictions with "
        "rule-based checks for links, domains, senders and email headers."
    )

    st.caption(
        "PhishGuard is a decision-support tool, not a guarantee. "
        "Treat unexpected requests for passwords, payments or sensitive "
        "information with caution."
    )