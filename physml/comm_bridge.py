"""Stage 143 — CommBridge: digital communication tool handlers.

Wires real communication channels into the GoalEngine so Myco can
send messages when asked by the user.

Channels
--------
* **Email** — stdlib ``smtplib``; zero extra deps.  Configure via env vars or
  companion personalisation (``email_host``, ``email_user``, ``email_pass``).
* **SMS** — Twilio SDK (optional dep ``twilio``).  Falls back to a log-only
  stub when Twilio is not installed or credentials are absent.
* **Slack** — Incoming Webhook (zero extra deps; just ``urllib``).
* **WhatsApp Web** — Playwright browser automation (drives web.whatsapp.com).
* **System notification** — plyer desktop notification as a last-resort channel.

All channels are gated: if credentials are missing the handler returns a
clear error message rather than silently failing.

Configuration
-------------
Set credentials via environment variables or companion.personalise():

    MYCO_EMAIL_HOST   smtp.gmail.com  (or outlook.office365.com, etc.)
    MYCO_EMAIL_PORT   587
    MYCO_EMAIL_USER   you@gmail.com
    MYCO_EMAIL_PASS   your-app-password

    TWILIO_ACCOUNT_SID  ACxxxxxx
    TWILIO_AUTH_TOKEN   xxxxxxxx
    TWILIO_FROM_NUMBER  +15551234567

    MYCO_SLACK_WEBHOOK  https://hooks.slack.com/services/...
"""

from __future__ import annotations

import json
import os
import re
import smtplib
import urllib.request
import urllib.parse
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Any, Optional

from physml._log import get_logger

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class CommResult:
    """Outcome of a communication action."""

    def __init__(self, success: bool, channel: str, message: str = "") -> None:
        self.success = success
        self.channel = channel
        self.message = message

    def __str__(self) -> str:
        status = "sent" if self.success else "failed"
        return f"[{self.channel}] {status}: {self.message}"


# ---------------------------------------------------------------------------
# CommBridge
# ---------------------------------------------------------------------------


class CommBridge:
    """Unified messaging bridge for GoalEngine step dispatch.

    Parameters
    ----------
    companion : MyceliumCompanion or None
        Provides browser_agent and notifier.
    email_host : str, optional
        SMTP host (env: MYCO_EMAIL_HOST).
    email_port : int, optional
        SMTP port (env: MYCO_EMAIL_PORT, default 587).
    email_user : str, optional
        SMTP username (env: MYCO_EMAIL_USER).
    email_pass : str, optional
        SMTP password or app-password (env: MYCO_EMAIL_PASS).
    slack_webhook : str, optional
        Slack Incoming Webhook URL (env: MYCO_SLACK_WEBHOOK).
    """

    def __init__(
        self,
        companion: Any = None,
        email_host: Optional[str] = None,
        email_port: int = 587,
        email_user: Optional[str] = None,
        email_pass: Optional[str] = None,
        slack_webhook: Optional[str] = None,
    ) -> None:
        self._companion = companion
        self.email_host = email_host or os.environ.get("MYCO_EMAIL_HOST", "")
        self.email_port = int(os.environ.get("MYCO_EMAIL_PORT", str(email_port)))
        self.email_user = email_user or os.environ.get("MYCO_EMAIL_USER", "")
        self.email_pass = email_pass or os.environ.get("MYCO_EMAIL_PASS", "")
        self.slack_webhook = slack_webhook or os.environ.get("MYCO_SLACK_WEBHOOK", "")

    # ------------------------------------------------------------------
    # Email
    # ------------------------------------------------------------------

    def send_email(
        self,
        to: str,
        subject: str,
        body: str,
        html: bool = False,
    ) -> CommResult:
        """Send an email via SMTP.

        Uses TLS (STARTTLS on port 587) or SSL (port 465) automatically.

        Parameters
        ----------
        to : str
            Recipient email address.
        subject : str
            Email subject.
        body : str
            Email body (plain text or HTML).
        html : bool
            If True, body is sent as text/html.
        """
        if not all([self.email_host, self.email_user, self.email_pass]):
            return CommResult(
                False, "email",
                "Email not configured. Set MYCO_EMAIL_HOST / USER / PASS env vars."
            )
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.email_user
            msg["To"] = to
            part = MIMEText(body, "html" if html else "plain")
            msg.attach(part)

            if self.email_port == 465:
                with smtplib.SMTP_SSL(self.email_host, self.email_port) as server:
                    server.login(self.email_user, self.email_pass)
                    server.sendmail(self.email_user, to, msg.as_string())
            else:
                with smtplib.SMTP(self.email_host, self.email_port) as server:
                    server.ehlo()
                    server.starttls()
                    server.login(self.email_user, self.email_pass)
                    server.sendmail(self.email_user, to, msg.as_string())

            _logger.info("CommBridge: email sent to %s — %r", to, subject)
            return CommResult(True, "email", f"Email sent to {to}: {subject!r}")
        except Exception as exc:
            _logger.warning("CommBridge: email failed: %s", exc)
            return CommResult(False, "email", str(exc))

    def parse_and_send_email(self, description: str) -> str:
        """Parse a natural-language step description and send an email.

        Recognises patterns like:
        - "send email to foo@bar.com subject 'Hi' body 'Message'"
        - "email alice@example.com: Hi there, just checking in"
        """
        # Extract email address
        emails = re.findall(r"[\w.\-+]+@[\w.\-]+\.[a-z]{2,}", description, re.I)
        if not emails:
            return "No email address found in step description."
        to = emails[0]

        # Extract subject
        subj_m = re.search(r"subject[:\s]+['\"]?([^'\"]+)['\"]?", description, re.I)
        subject = subj_m.group(1).strip() if subj_m else "Message from Myco"

        # Extract body
        body_m = re.search(r"body[:\s]+['\"]?(.+)['\"]?$", description, re.I | re.S)
        if body_m:
            body = body_m.group(1).strip()
        else:
            # Everything after the email address is the body
            idx = description.find(to) + len(to)
            body = description[idx:].strip().lstrip(":,;— ").strip()
            if not body:
                body = description

        result = self.send_email(to, subject, body)
        return str(result)

    # ------------------------------------------------------------------
    # SMS via Twilio
    # ------------------------------------------------------------------

    def send_sms(self, to: str, body: str) -> CommResult:
        """Send an SMS via Twilio.

        Requires: ``pip install twilio`` and env vars
        TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM_NUMBER.
        """
        sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
        token = os.environ.get("TWILIO_AUTH_TOKEN", "")
        from_num = os.environ.get("TWILIO_FROM_NUMBER", "")

        if not all([sid, token, from_num]):
            return CommResult(
                False, "sms",
                "SMS not configured. Set TWILIO_ACCOUNT_SID / AUTH_TOKEN / FROM_NUMBER."
            )
        try:
            from twilio.rest import Client  # type: ignore
            client = Client(sid, token)
            message = client.messages.create(body=body, from_=from_num, to=to)
            _logger.info("CommBridge: SMS sent to %s (sid=%s)", to, message.sid)
            return CommResult(True, "sms", f"SMS sent to {to}")
        except ImportError:
            return CommResult(
                False, "sms",
                "Twilio SDK not installed. Run: pip install twilio"
            )
        except Exception as exc:
            _logger.warning("CommBridge: SMS failed: %s", exc)
            return CommResult(False, "sms", str(exc))

    def parse_and_send_sms(self, description: str) -> str:
        """Parse a NL step and send SMS."""
        phones = re.findall(r"\+?\d[\d\s\-().]{7,}\d", description)
        if not phones:
            return "No phone number found in step description."
        to = re.sub(r"[\s\-()]", "", phones[0])
        # Body is everything after the phone number
        idx = description.find(phones[0]) + len(phones[0])
        body = description[idx:].strip().lstrip(":,;— ").strip() or description
        return str(self.send_sms(to, body))

    # ------------------------------------------------------------------
    # Slack webhook
    # ------------------------------------------------------------------

    def send_slack(self, message: str, channel: str = "") -> CommResult:
        """Post a message to Slack via Incoming Webhook."""
        webhook = self.slack_webhook
        if not webhook:
            return CommResult(
                False, "slack",
                "Slack webhook not configured. Set MYCO_SLACK_WEBHOOK env var."
            )
        try:
            payload = {"text": message}
            if channel:
                payload["channel"] = channel
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                webhook, data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
            _logger.info("CommBridge: Slack message sent")
            return CommResult(True, "slack", f"Slack message sent: {message[:60]!r}")
        except Exception as exc:
            _logger.warning("CommBridge: Slack failed: %s", exc)
            return CommResult(False, "slack", str(exc))

    # ------------------------------------------------------------------
    # WhatsApp Web (browser automation)
    # ------------------------------------------------------------------

    def send_whatsapp(self, contact_name: str, message: str) -> CommResult:
        """Send a WhatsApp message via web.whatsapp.com.

        Requires: Playwright (``pip install playwright && playwright install chromium``)
        and that the user has previously logged into WhatsApp Web in the
        non-headless browser (first run opens the QR-code page).

        Parameters
        ----------
        contact_name : str
            Display name of the contact as shown in WhatsApp.
        message : str
            Message text to send.
        """
        if self._companion is None or self._companion.browser_agent is None:
            return CommResult(False, "whatsapp", "Browser agent not available.")

        ba = self._companion.browser_agent
        if not ba.available:
            return CommResult(
                False, "whatsapp",
                "Playwright not installed. Run: pip install playwright && playwright install chromium"
            )
        try:
            # Open WhatsApp Web (non-headless for QR scan on first use)
            if not ba.goto("https://web.whatsapp.com", wait_until="networkidle"):
                return CommResult(False, "whatsapp", "Could not load WhatsApp Web.")

            import time
            time.sleep(3)

            # Search for contact
            search = ba._page.locator('div[data-testid="chat-list-search"]')
            if search.count():
                search.click()
                search.fill(contact_name)
                time.sleep(1.5)
                # Click first result
                first = ba._page.locator('div[data-testid="cell-frame-container"]').first
                if first.count():
                    first.click()
                    time.sleep(1)

            # Type and send message
            msg_box = ba._page.locator('div[data-testid="conversation-compose-box-input"]')
            if msg_box.count():
                msg_box.click()
                msg_box.fill(message)
                ba._page.keyboard.press("Enter")
                time.sleep(0.5)
                _logger.info("CommBridge: WhatsApp message sent to %r", contact_name)
                return CommResult(True, "whatsapp", f"WhatsApp message sent to {contact_name!r}")
            return CommResult(False, "whatsapp", "Could not find message input box.")
        except Exception as exc:
            _logger.warning("CommBridge: WhatsApp failed: %s", exc)
            return CommResult(False, "whatsapp", str(exc))

    # ------------------------------------------------------------------
    # Parse-and-dispatch: called from GoalEngine handler
    # ------------------------------------------------------------------

    def dispatch(self, description: str) -> str:
        """Route a NL step description to the right channel.

        Routing heuristic:
        - Contains @-address and email/mail keywords → email
        - Contains phone number and sms/text/message keywords → SMS
        - Contains whatsapp/wa → WhatsApp
        - Contains slack → Slack
        - Fallback → notification via Notifier
        """
        low = description.lower()
        has_email = bool(re.search(r"[\w.\-+]+@[\w.\-]+\.[a-z]{2,}", description))
        has_phone = bool(re.search(r"\+?\d[\d\s\-().]{7,}\d", description))

        if has_email and any(k in low for k in ("email", "mail", "send", "message")):
            return self.parse_and_send_email(description)

        if any(k in low for k in ("whatsapp", "whats app", " wa ")):
            # Extract contact name and message
            m = re.search(r"(?:to|send to|message)\s+([a-zA-Z ]+?)(?:\s+[\"'](.+)[\"']|:\s*(.+))?$",
                          description, re.I)
            contact = m.group(1).strip() if m else "contact"
            msg = (m.group(2) or m.group(3) or description).strip()
            return str(self.send_whatsapp(contact, msg))

        if any(k in low for k in ("slack", "channel", "webhook")):
            m = re.search(r"(?:message|say|post)[:\s]+(.+)$", description, re.I)
            msg = m.group(1).strip() if m else description
            return str(self.send_slack(msg))

        if has_phone and any(k in low for k in ("sms", "text", "message")):
            return self.parse_and_send_sms(description)

        # Fallback: desktop notification
        if self._companion and getattr(self._companion, "notifier", None):
            self._companion.notifier.send("Myco Message", description)
            return f"Sent as desktop notification: {description[:80]}"

        return f"No messaging channel matched for: {description[:80]}"

    def status(self) -> dict:
        """Return configuration status for each channel."""
        return {
            "email": bool(self.email_host and self.email_user and self.email_pass),
            "sms_twilio": bool(
                os.environ.get("TWILIO_ACCOUNT_SID") and
                os.environ.get("TWILIO_AUTH_TOKEN")
            ),
            "slack": bool(self.slack_webhook),
            "whatsapp": (
                self._companion is not None and
                getattr(self._companion, "browser_agent", None) is not None and
                getattr(self._companion.browser_agent, "available", False)
            ),
        }
