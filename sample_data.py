"""
Sample spam/ham email data for demonstration
"""

SAMPLE_EMAILS = [
    # Spam emails
    ("URGENT! You have won $1000000! Click here to claim your prize now!", "spam"),
    ("Congratulations! You've been selected for a special offer. Call now!", "spam"),
    ("FREE! Get rich quick scheme. Limited time offer!", "spam"),
    ("WINNER! You have won a lottery! Send your details immediately!", "spam"),
    ("Cheap viagra! Buy now with 90% discount!", "spam"),
    ("Make money fast! Work from home opportunity!", "spam"),
    ("URGENT: Your account will be closed. Click here to verify!", "spam"),
    ("Get a loan with no credit check required!", "spam"),
    ("You have inherited $5 million from a distant relative!", "spam"),
    ("Hot singles in your area want to meet you!", "spam"),
    ("Free iPhone! Just pay shipping and handling!", "spam"),
    ("Lose 30 pounds in 30 days guaranteed!", "spam"),
    ("Your computer is infected! Download our antivirus now!", "spam"),
    ("Congratulations! You qualify for a $10000 credit card!", "spam"),
    ("Act now! Limited time offer expires today!", "spam"),
    
    # Ham emails
    ("Hi John, can we schedule a meeting for tomorrow at 2 PM?", "ham"),
    ("Thank you for your presentation today. It was very informative.", "ham"),
    ("Please find the attached report for your review.", "ham"),
    ("Reminder: Team meeting scheduled for Friday at 10 AM.", "ham"),
    ("Your order has been shipped and will arrive in 3-5 business days.", "ham"),
    ("Welcome to our newsletter! Here's this week's update.", "ham"),
    ("Your appointment is confirmed for Monday at 3 PM.", "ham"),
    ("Happy birthday! Hope you have a wonderful day.", "ham"),
    ("The project deadline has been extended to next Friday.", "ham"),
    ("Please review the contract and let me know your thoughts.", "ham"),
    ("Your subscription will expire in 30 days. Renew now to continue.", "ham"),
    ("Thank you for your purchase. Your receipt is attached.", "ham"),
    ("The weather forecast shows rain tomorrow. Don't forget your umbrella.", "ham"),
    ("Your flight is on time. Gate information will be available 2 hours before departure.", "ham"),
    ("Congratulations on your promotion! Well deserved.", "ham"),
    ("The conference has been rescheduled to next month due to venue issues.", "ham"),
    ("Your password has been successfully changed.", "ham"),
    ("Monthly report is due by the end of this week.", "ham"),
    ("Thank you for attending our webinar. The recording is now available.", "ham"),
    ("Your library books are due for return by Friday.", "ham"),
]

def get_sample_data():
    """Return sample emails and labels as lists"""
    emails = [email[0] for email in SAMPLE_EMAILS]
    labels = [email[1] for email in SAMPLE_EMAILS]
    return emails, labels