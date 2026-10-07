"""Demo emails for the live demo (all fake / made-up)."""
DEMOS = {
    "Bank KYC scam (phishing)": dict(
        sender="alerts@sbi-secure-kyc.com", reply_to="helpdesk@gmail.com",
        subject="URGENT: Your account will be suspended - verify KYC immediately",
        body="Dear customer,\nYour account has been flagged and will be suspended within 24 hours.\n"
             "Confirm your password, OTP and Aadhaar details immediately to avoid closure.\n"
             "Click here: http://192.168.44.12/login/verify\nAct now, limited time."),
    "Parcel fee scam (phishing)": dict(
        sender="support@parcel-delivery-update.co", reply_to="",
        subject="Your parcel is on hold - pay customs fee",
        body="Your parcel could not be delivered. Confirm your card number and PIN and pay the small fee to release it.\n"
             "Pay now: http://bit.ly/parcel-release-now\nThis is urgent, the parcel will be returned."),
    "Account blocked scam (phishing)": dict(
        sender="security@netfIix-billing.com", reply_to="billing.help@gmail.com",
        subject="Payment failed - account will be blocked",
        body="Your account will be blocked immediately. Verify your account and update your password and card number.\n"
             "Update here: http://tinyurl.com/update-billing\nUrgent - act now."),
    "Fake job offer (phishing)": dict(
        sender="hr@top-jobs-offer.com", reply_to="",
        subject="Congratulations! Job selected - pay Rs 999 registration",
        body="You are selected for a work-from-home job. To confirm, submit your PAN card, Aadhaar and pay the registration fee.\n"
             "Confirm within 2 hours: http://bit.ly/job-confirm"),
    "Meeting notice (legitimate)": dict(
        sender="office@company.co.in", reply_to="",
        subject="Team meeting moved to Thursday 3 PM",
        body="Hi all,\nThe weekly team meeting has moved to Thursday at 3 PM in the conference room.\n"
             "Please bring your updates. The agenda is attached in the shared folder.\nThanks,\nAdmin Office"),
    "Order confirmation (legitimate)": dict(
        sender="orders@shop.example.in", reply_to="",
        subject="Your order has been shipped",
        body="Hello,\nYour order #48213 has shipped and should arrive on Friday.\nYou can track it from the Orders page in your account.\nThank you for shopping with us."),
}
