"""
Builds a modern SMS/text spam+ham augmentation set to merge with the
old 2005 UCI SMS Spam Collection. Targets exactly the gap diagnosed:
phishing links, fake prizes, fake discounts, fake bank alerts,
fake job offers, fake gifts -- the categories that scored <50%
confidence on the original model.

Generated via templated variation (not scraped/copied from any source),
covering realistic modern phrasing patterns in English (India-relevant
context included, e.g. INR amounts).
"""
import csv
import random

random.seed(42)

# ---------------- SPAM TEMPLATES ----------------

prize_amounts = ["₹50,000", "₹1,00,000", "₹25,000", "$500", "$1000", "₹10,000", "₹75,000"]
fake_domains = ["www.luckyprizes.in", "www.superdeal.in", "www.fakebanklink.com",
                 "www.easycash.com", "www.claimnow.in", "bit.ly/3xWinBig",
                 "www.megaoffers.co", "www.quickreward.net", "www.instawin.in",
                 "secure-update-portal.com", "www.giftclaim.in", "www.cashbonus.co.in"]
banks = ["SBI", "HDFC", "ICICI", "Axis Bank", "your bank", "Paytm", "PhonePe"]
brands = ["Amazon", "Flipkart", "Myntra", "our store", "all products", "electronics"]

spam_templates = [
    "Congratulations! You have won {amt}! Click here to claim your prize now: {url}",
    "WINNER! You've been selected to receive {amt} cash reward. Visit {url} to claim before it expires.",
    "Limited Offer! Get {pct}% off on {brand}. Visit our website immediately: {url}",
    "FLASH SALE: {pct}% discount today only on {brand}! Shop now: {url}",
    "Your {bank} account has been suspended! Verify now at {url}",
    "Security Alert: unusual login detected on your {bank} account. Verify immediately: {url}",
    "Dear customer, your account will be blocked in 24 hours. Update KYC now at {url}",
    "Earn money from home! No experience required. Sign up today: {url}",
    "Work from home and earn {amt} per month. No investment needed. Register: {url}",
    "You've been selected for a special gift. Reply 'YES' to claim it!",
    "Congratulations! You are eligible for a free gift box. Reply YES now to claim.",
    "URGENT: your parcel is on hold due to unpaid customs fee. Pay now at {url}",
    "Your {brand} order is delayed. Confirm your address immediately: {url}",
    "Last chance! Your {amt} reward expires today. Claim at {url}",
    "Hi, this is {bank} fraud team. Your card has been blocked. Call this number immediately to unblock.",
    "You have an unclaimed refund of {amt}. Click {url} to receive it instantly.",
    "Free recharge of {amt} waiting for you! Click {url} now to activate.",
    "Act now! Only few hours left to claim your {amt} cashback. Visit {url}",
    "Your loan of {amt} is pre-approved. Apply now with zero documents: {url}",
    "Get rich quick! Invest {amt} today and double it in a week. Join now: {url}",
]

# ---------------- HAM TEMPLATES (modern, everyday) ----------------

names = ["Ramesh", "Priya", "Arjun", "Divya", "Karthik", "Anita", "Suresh", "Lakshmi"]
places = ["Mount Road", "the railway station", "the office", "the mall", "MG Road", "the clinic"]
cities = ["Chennai", "Bangalore", "Hyderabad", "Mumbai", "Coimbatore", "Madurai"]

ham_templates = [
    "Hi {name}, are we meeting for lunch at {time} today at the restaurant on {place}?",
    "Dear customer, your electricity bill for this month is ₹{bill}. Please pay by {day}th to avoid late fees.",
    "Reminder: Your dentist appointment is scheduled for {day}th at {time}.",
    "Happy Birthday! Wishing you a wonderful year ahead filled with joy and success.",
    "The train to {city} departs at {time}. Please be at the station 30 minutes early.",
    "Hey, can you send me the report before {time} today?",
    "Your order has been shipped and will arrive by {day}th. Track it on our app.",
    "Mom, I'll be home by {time}, don't wait for dinner.",
    "Meeting rescheduled to {time} tomorrow in the conference room.",
    "Thanks for coming to the party yesterday, it meant a lot to us.",
    "Your OTP for login is 482913. Do not share this with anyone.",
    "Your prescription is ready for pickup at the pharmacy near {place}.",
    "Just landed in {city}, will call you once I reach the hotel.",
    "Can you pick up milk and bread on your way home?",
    "Your monthly statement is now available to view in the app.",
    "Don't forget the parent-teacher meeting at school on {day}th.",
    "Let's catch up this weekend, it's been a while.",
    "The plumber will visit your flat between {time} and {time2} tomorrow.",
    "Your gym membership renewal is due on {day}th this month.",
    "Congrats on your promotion! So happy for you, let's celebrate soon.",
]

times = ["10:00 AM", "1 PM", "5:30 PM", "9:00 AM", "2:30 PM", "6:00 PM", "11:00 AM"]


def fill(template):
    return template.format(
        amt=random.choice(prize_amounts),
        url=random.choice(fake_domains),
        pct=random.choice(["50", "60", "70", "80", "40"]),
        brand=random.choice(brands),
        bank=random.choice(banks),
        name=random.choice(names),
        time=random.choice(times),
        time2=random.choice(times),
        place=random.choice(places),
        city=random.choice(cities),
        day=random.choice(["15", "20", "22", "25", "28", "5", "10"]),
        bill=random.choice(["1,250", "2,350", "1,890", "3,100", "980"]),
    )


def generate(templates, n_per_template=15):
    out = set()
    for t in templates:
        for _ in range(n_per_template):
            out.add(fill(t))
    return list(out)


if __name__ == "__main__":
    spam_msgs = generate(spam_templates, n_per_template=15)
    ham_msgs = generate(ham_templates, n_per_template=15)

    print(f"Generated {len(spam_msgs)} unique spam, {len(ham_msgs)} unique ham")

    with open("augmented_modern.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["v1", "v2"])
        for m in spam_msgs:
            writer.writerow(["spam", m])
        for m in ham_msgs:
            writer.writerow(["ham", m])

    print("Saved to augmented_modern.csv")
