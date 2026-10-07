import re
import string
from email import message_from_bytes, policy
from html import escape
from html.parser import HTMLParser

import numpy as np
from scipy.sparse import hstack, csr_matrix

try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
HTML_PATTERN = re.compile(r"<.*?>")
EMAIL_PATTERN = re.compile(r"[\w\.-]+@([\w\.-]+)")
DOMAIN_IN_URL_PATTERN = re.compile(r"https?://(?:www\.)?([^/\s]+)|www\.([^/\s]+)")

URGENCY_WORDS = [
    "urgent", "verify", "suspend", "immediately", "password",
    "click", "confirm", "account", "act now", "limited time",
]
FREE_EMAIL_DOMAINS = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com"}
META_COLS = ["url_count", "urgency_word_count", "text_length",
             "domain_count", "suspicious_domain"]


def count_urls(text):
    return len(URL_PATTERN.findall(str(text)))


def count_urgency_words(text):
    t = str(text).lower()
    return sum(1 for w in URGENCY_WORDS if w in t)


def extract_domains(text):
    text = str(text)
    domains = set()
    for m in EMAIL_PATTERN.findall(text):
        domains.add(m.lower())
    for a, b in DOMAIN_IN_URL_PATTERN.findall(text):
        d = (a or b).lower()
        if d:
            domains.add(d)
    return domains


def has_suspicious_domain(text):
    domains = extract_domains(text)
    if not domains:
        return 0
    return int(all(d in FREE_EMAIL_DOMAINS for d in domains))


def domain_count(text):
    return len(extract_domains(text))


def clean_text(text):
    text = str(text).lower()
    text = HTML_PATTERN.sub(" ", text)
    text = URL_PATTERN.sub(" URLTOKEN ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(tokens)


def meta_features(text):
    return [
        count_urls(text), count_urgency_words(text), len(str(text)),
        domain_count(text), has_suspicious_domain(text),
    ]

def vectorize(art, text):
    """raw email text -> the same sparse matrix the models were trained on."""
    tf = art["tfidf"].transform([clean_text(text)])
    meta = art["scaler"].transform(np.array([meta_features(text)], dtype=float))
    return hstack([tf, csr_matrix(meta)]).tocsr(), tf


def predict(art, text):
    X, tf = vectorize(art, text)
    p_rf = float(art["rf"].predict_proba(X)[0, 1])
    p_lr = float(art["lr"].predict_proba(X)[0, 1])
    return {"ml_prob": (p_rf + p_lr) / 2, "p_rf": p_rf, "p_lr": p_lr, "tf": tf}


def explain(art, tf, top_n=10):
    """Per-email word contributions from the Logistic Regression weights.
    contribution = coefficient x tf-idf value  (positive => pushes toward phishing)."""
    names = art["tfidf"].get_feature_names_out()
    coef = art["lr"].coef_[0][: len(names)]
    contrib = tf.toarray()[0] * coef
    order = np.argsort(contrib)
    phish = [(names[i], float(contrib[i])) for i in order[::-1][:top_n] if contrib[i] > 0]
    legit = [(names[i], float(contrib[i])) for i in order[:top_n] if contrib[i] < 0]
    return phish, legit


def highlight_html(text, phish_terms):
    """Return HTML of the email with phishing-driving words marked in red."""
    text = tidy(text)
    words = {w for term in phish_terms for w in term.split() if len(w) > 2}
    out, last = [], 0
    for m in re.finditer(r"[A-Za-z0-9']+", text):
        out.append(escape(text[last:m.start()]))
        w = m.group(0)
        if w.lower() in words:
            out.append(f'<mark style="background:#ffd6d6;color:#8b0000;'
                       f'padding:0 2px;border-radius:3px">{escape(w)}</mark>')
        else:
            out.append(escape(w))
        last = m.end()
    out.append(escape(text[last:]))
    return '<div style="white-space:pre-wrap;line-height:1.6">' + "".join(out) + "</div>"

BRANDS = ["paypal", "amazon", "microsoft", "google", "apple", "netflix", "sbi",
          "hdfc", "icici", "paytm", "flipkart", "irctc", "facebook", "instagram",
          "whatsapp", "linkedin", "axis", "npci", "uidai", "incometax"]
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
              "buff.ly", "cutt.ly", "rb.gy", "shorturl.at", "tiny.cc"}
IP_HOST = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s", "4": "a", "$": "s"})


def _edit1(a, b):
    """True if a and b differ by exactly one edit (insert/delete/replace)."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    s, l = (a, b) if len(a) < len(b) else (b, a)
    return any(l[:i] + l[i + 1:] == s for i in range(len(l)))


def _host(url):
    u = url.strip().lower()
    if u.startswith(("mailto:", "tel:", "sms:", "#", "javascript:")):
        return ""
    m = re.match(r"(?:https?://)?(?:www\.)?([^/:?#\s]+)", u)
    return m.group(1).rstrip(").,;>") if m else ""


def _brand_issue(host):
    """Return (brand, why) if the host imitates a known brand, else None."""
    if _under(host, TRACKER_DOMAINS | INFRA_DOMAINS | OFFICIAL_DOMAINS):
        return None
    labels = host.split(".")
    for label in labels[:-1] or labels:
        clean = label.replace("-", "")
        for b in BRANDS:
            if label == b:
                return None  
            for tok in {clean, *label.split("-")}:
                if tok and (tok.translate(LEET) == b or (_edit1(tok, b) and len(b) >= 6)) and tok != b:
                    return b, f"'{host}' looks like a misspelling of {b}"
            if (b in label.split("-") and label != b) or (b in label and len(b) >= 6):
                return b, f"'{host}' uses the name {b} but is not the official site"
    return None


INVISIBLE = re.compile(r"[\u200b-\u200f\u2060\u00ad\ufeff\u034f]")


def tidy(text):
    """Remove zero-width junk, collapse spaces, drop blank lines and indentation."""
    text = INVISIBLE.sub("", str(text)).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)


class _LinkParser(HTMLParser):
    SKIP = {"style", "script", "head", "title"}
    BREAK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table"}

    def __init__(self):
        super().__init__()
        self.links, self._href, self._txt, self.text, self._skip = [], None, [], [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        if tag in self.BREAK:
            self.text.append("\n")
        if tag == "a":
            self._href, self._txt = dict(attrs).get("href"), []

    def handle_data(self, data):
        if self._skip:
            return
        self.text.append(data)
        if self._href is not None:
            self._txt.append(data)

    def handle_endtag(self, tag):
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        if tag in self.BREAK:
            self.text.append("\n")
        if tag == "a" and self._href is not None:
            self.links.append((tidy("".join(self._txt)), self._href))
            self._href = None


def parse_message(msg):
    """email.message object -> dict(subject, sender, reply_to, auth, body, links)."""
    body_part = msg.get_body(preferencelist=("plain", "html"))
    body, links = "", []
    if body_part is not None:
        content = body_part.get_content()
        if body_part.get_content_type() == "text/html":
            p = _LinkParser()
            p.feed(content)
            links = p.links
            body = tidy("".join(p.text))
        else:
            body = tidy(content)
    if not links:
        html_part = msg.get_body(preferencelist=("html",))
        if html_part is not None:
            p = _LinkParser()
            p.feed(html_part.get_content())
            links = p.links
    return {
        "subject": tidy(msg.get("Subject", "") or ""),
        "sender": str(msg.get("From", "") or ""),
        "reply_to": str(msg.get("Reply-To", "") or ""),
        "auth": " ".join(str(x) for x in msg.get_all("Authentication-Results", [])),
        "body": body,
        "links": links,
    }


def parse_eml(raw_bytes):
    return parse_message(message_from_bytes(raw_bytes, policy=policy.default))


def iter_local_emails(path, limit=500):
    """Yield email dicts from a folder of .eml files and/or .mbox files (e.g. Gmail Takeout)."""
    import mailbox
    import os
    n = 0
    for root, _, files in os.walk(path):
        for fn in sorted(files):
            full = os.path.join(root, fn)
            try:
                if fn.lower().endswith(".eml"):
                    with open(full, "rb") as fh:
                        yield parse_eml(fh.read())
                        n += 1
                elif fn.lower().endswith(".mbox"):
                    box = mailbox.mbox(full, factory=lambda f: __import__("email").message_from_binary_file(f, policy=policy.default))
                    for m in box:
                        yield parse_message(m)
                        n += 1
                        if n >= limit:
                            return
            except Exception:  
                continue
            if n >= limit:
                return


def auth_status(auth):
    """Parse an Authentication-Results header -> {'spf': 'pass'|'fail'|'none', ...}."""
    out = {}
    for c in ("spf", "dkim", "dmarc"):
        found = re.findall(rf"\b{c}=(\w+)", str(auth).lower())
        if "pass" in found:
            out[c] = "pass"
        elif any(f in ("fail", "softfail", "permerror") for f in found):
            out[c] = "fail"
        elif found:
            out[c] = "none"
    return out

OFFICIAL_DOMAINS = {
    "linkedin.com", "google.com", "microsoft.com", "amazon.com", "amazon.in", "apple.com",
    "paypal.com", "netflix.com", "facebookmail.com", "facebook.com", "instagram.com",
    "github.com", "sbi.co.in", "onlinesbi.sbi", "hdfcbank.com", "icicibank.com",
    "axisbank.com", "paytm.com", "flipkart.com", "irctc.co.in", "naukri.com",
    "coursera.org", "unstop.com", "devfolio.co", "microsoftonline.com", "youtube.com",
}


def trust_signals(email, flags):
    """Return (list_of_positive_messages, trusted_bool)."""
    if any(sev == "high" for sev, _ in flags):
        return [], False
    dom = (EMAIL_PATTERN.findall(email.get("sender", "")) or [""])[0].lower()
    st = auth_status(email.get("auth", ""))
    passed = [k.upper() for k, v in st.items() if v == "pass"]
    auth_ok = ("dkim" in [p.lower() for p in passed] or "spf" in [p.lower() for p in passed])
    official = bool(dom) and any(dom == d or dom.endswith("." + d) for d in OFFICIAL_DOMAINS)
    pos = []
    if auth_ok:
        pos.append("Sender authentication passed (" + ", ".join(passed) + ")")
    if official:
        pos.append(f"Sender domain {dom} is a known official domain")
    return pos, (auth_ok and official)


TRACK_PREFIX = {"mx", "click", "clicks", "link", "links", "track", "trk", "email", "e", "go", "url", "t", "r", "mail"}
SECOND_LEVEL = {"co", "com", "org", "net", "ac", "gov", "edu", "nic"}


def reg_domain(host):
    """mail.bserc.org -> bserc.org ; x.co.in -> x.co.in (registrable domain)."""
    parts = host.lower().strip(".").split(".")
    if len(parts) >= 3 and len(parts[-1]) == 2 and parts[-2] in SECOND_LEVEL:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def _under(host, domains):
    return any(host == d or host.endswith("." + d) for d in domains)

TRACKER_DOMAINS = {
    "awstrack.me", "amazonses.com", "customer.io", "sendgrid.net", "sendgrid.com",
    "mailchimp.com", "list-manage.com", "mandrillapp.com", "mailgun.org", "mailjet.com",
    "sparkpostmail.com", "hubspotlinks.com", "hubspotemail.net", "sendinblue.com",
    "brevo.com", "constantcontact.com", "rs6.net", "klclick.com", "klaviyomail.com",
    "exct.net", "iterable.com", "sailthru.com", "braze.com", "mkt.com", "cmail19.com",
    "createsend.com", "campaign-archive.com", "technolutions.net", "aweber.com", "salesforce.com", "protection.outlook.com",
    "urldefense.com", "proofpoint.com", "mimecast.com", "google.com", "gstatic.com",
}
INFRA_DOMAINS = {
    "amazonaws.com", "cloudfront.net", "googleusercontent.com", "googleapis.com",
    "azureedge.net", "windows.net", "licdn.com", "fbcdn.net", "apple.news",
}

DOMAIN_TEXT = re.compile(
    r"^(?:https?://)?(?:www\.)?([a-z0-9-]+(?:\.[a-z0-9-]+)*\.[a-z]{2,24})(?:[/?#]\S*)?$", re.I)

_HIGH_SECRET = r"otp|cvv|pin|password|card number"
_LOW_SECRET = r"aadhaar|aadhar|pan card|kyc"
ASK = re.compile(
    r"\b(?:enter|share|send|provide|confirm|verify|update|submit|reply with|give|complete)\b"
    r"[^.!?]{0,40}?\b(?P<what>" + _HIGH_SECRET + "|" + _LOW_SECRET + r")\b", re.I)
NEGATION = re.compile(r"\b(?:never|ever|not|don'?t|won'?t|nobody|no one)\s*$", re.I)


BAIT = re.compile(
    r"\b(?:bitcoin|btc|crypto(?:currency)?|jackpot|lottery|giveaway|winner|free money|"
    r"you(?:'ve| have) won|double your|guaranteed returns?|claim your (?:prize|reward|bonus)|"
    r"earn (?:rs\.?|\$|₹)?\s?\d+[^.]{0,20}(?:per|a) day)\b", re.I)


def asks_for_secrets(text):
    """Return 'high' / 'low' / None. Ignores warnings such as 'never share your OTP'."""
    level = None
    for m in ASK.finditer(text):
        before = text[max(0, m.start() - 25):m.start()].rstrip()
        if NEGATION.search(before):
            continue
        what = m.group("what").lower()
        if re.fullmatch(_HIGH_SECRET, what):
            return "high"
        level = "low"
    return level


def rule_flags(email):
    """email: dict(subject, sender, reply_to, auth, body, links).
    Returns list of (severity, message). Similar findings are merged into ONE flag,
    so a newsletter with 40 tracked links cannot pile up 40 penalties."""
    flags = []
    body = email.get("body", "")
    sender = email.get("sender", "")
    text_all = f'{email.get("subject", "")} {body}'

    sender_dom = (EMAIL_PATTERN.findall(sender) or [""])[0].lower()
    reply_dom = (EMAIL_PATTERN.findall(email.get("reply_to", "")) or [""])[0].lower()
    sender_reg = reg_domain(sender_dom) if sender_dom else ""

    if sender_dom and reply_dom and reg_domain(reply_dom) != sender_reg:
        flags.append(("high", f"Reply-To domain ({reply_dom}) differs from sender domain ({sender_dom})"))

    failed = [k.upper() for k, v in auth_status(email.get("auth", "")).items() if v == "fail"]
    if failed:
        flags.append(("high", "Sender authentication FAILED (" + ", ".join(failed) + "): the sender may be spoofed"))

    if sender_dom:
        issue = _brand_issue(sender_dom)
        if issue:
            flags.append(("high", f"Sender domain: {issue[1]}"))
        if sender_dom in FREE_EMAIL_DOMAINS and any(b in text_all.lower() for b in BRANDS):
            flags.append(("low", f"Mentions a well-known brand but is sent from a free mailbox ({sender_dom})"))

    hrefs = [h for _, h in email.get("links", []) if h and h.lower().startswith(("http://", "https://"))]
    seen, ip_hosts, short, plain_http, brand = set(), [], [], [], []
    for u in URL_PATTERN.findall(text_all) + hrefs:
        h = _host(u)
        if not h or h in seen:
            continue
        seen.add(h)
        if _under(h, TRACKER_DOMAINS | INFRA_DOMAINS | OFFICIAL_DOMAINS):
            continue
        if IP_HOST.match(h):
            ip_hosts.append(h)
        elif h in SHORTENERS:
            short.append(h)
        else:
            if u.lower().startswith("http://"):
                plain_http.append(h)
            issue = _brand_issue(h)
            if issue:
                brand.append(issue[1])
    if ip_hosts:
        flags.append(("high", "Link goes to a raw IP address (" + ", ".join(ip_hosts[:3]) + ")"))
    if brand:
        more = f" (+{len(brand) - 1} more)" if len(brand) > 1 else ""
        flags.append(("high", "Link: " + brand[0] + more))
    if short:
        flags.append(("low", "Shortened link hides the real destination (" + ", ".join(short[:3]) + ")"))
    if plain_http:
        flags.append(("low", f"{len(plain_http)} link(s) not encrypted (http): " + ", ".join(plain_http[:3])))

    mism, mism_low = [], []
    for txt, href in email.get("links", []):
        if not href or not href.lower().startswith(("http://", "https://")):
            continue
        m = DOMAIN_TEXT.match(txt.strip())
        h_host = _host(href)
        if not m or not h_host:
            continue
        t_host = m.group(1).lower()
        if reg_domain(t_host) == reg_domain(h_host) or reg_domain(h_host) == sender_reg:
            continue
        if _under(h_host, TRACKER_DOMAINS):
            continue
        if h_host.split(".")[0] in TRACK_PREFIX:  
            mism_low.append((t_host, h_host))
            continue
        mism.append((t_host, h_host))
    if mism:
        t, h = mism[0]
        flags.append(("high", f"{len(mism)} link(s) show one web address but lead elsewhere, "
                              f"e.g. text '{t}' goes to '{h}'"))

    if mism_low and not mism:
        t, h = mism_low[0]
        flags.append(("low", f"{len(mism_low)} link(s) go through a tracking-style address, e.g. '{t}' via '{h}'"))

    bait = sorted({m.lower() for m in BAIT.findall(text_all)})
    if len(bait) >= 2:
        flags.append(("high", "Money / prize bait wording: " + ", ".join(bait[:4])))
    elif bait:
        flags.append(("low", "Money / prize bait wording: " + bait[0]))

    urg = [w for w in URGENCY_WORDS if w in text_all.lower()]
    if len(urg) >= 3:
        flags.append(("low", "Pressure language: " + ", ".join(urg)))
    secret = asks_for_secrets(text_all)
    if secret:
        flags.append((secret, "Asks you to enter/share sensitive details (OTP / PIN / password / KYC / Aadhaar)"))

    uniq, out = set(), []
    for f in flags:
        if f[1] not in uniq:
            uniq.add(f[1])
            out.append(f)
    return out


def rule_score(flags):
    """Noisy-OR: each distinct flag adds risk, but they never just add up to 100%."""
    keep = 1.0
    for sev, _ in flags:
        keep *= 1 - (0.35 if sev == "high" else 0.08)
    return 1 - keep

def risk_level(p):
    if p < 0.35:
        return "LOW"
    if p < 0.65:
        return "MEDIUM"
    return "HIGH"


ADVICE = {
    "LOW": ["No strong phishing signs found, but stay alert.",
            "If it asks you to act quickly or share details, verify with the sender another way."],
    "MEDIUM": ["Treat this email with suspicion.",
               "Do not click links or open attachments until you verify the sender.",
               "Contact the organisation using a number or website you already trust."],
    "HIGH": ["Do NOT click any link, open attachments or reply.",
             "Never share passwords, OTPs, PINs or card details by email.",
             "Report it (India: cybercrime.gov.in / 1930) and delete it."],
}


def analyze(art, email):
    """Full analysis of one email dict. Returns everything the UI needs."""
    text = tidy(f'{email.get("subject", "")}\n{email.get("body", "")}')
    pred = predict(art, text)
    phish, legit = explain(art, pred["tf"])
    flags = rule_flags(email)
    positives, trusted = trust_signals(email, flags)
    ml_raw = pred["ml_prob"]
    quiet = ((not flags) and count_urls(text) == 0 and count_urgency_words(text) < 2
             and pred["p_rf"] < 0.85)  
    if trusted:
        ml_used = ml_raw * 0.35
    elif quiet:
        ml_used = ml_raw * 0.4
    else:
        ml_used = ml_raw
    rscore = rule_score(flags)
    final = 1 - (1 - ml_used) * (1 - rscore)   
    return {
        "text": text, "ml_raw": ml_raw, "ml_prob": ml_used, "trusted": trusted,
        "p_rf": pred["p_rf"], "p_lr": pred["p_lr"],
        "quiet": quiet and not trusted, "rule_score": rscore, "final": final, "level": risk_level(final),
        "flags": flags, "positives": positives,
        "phish_terms": phish, "legit_terms": legit,
    }
