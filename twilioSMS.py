import os
from twilio.rest import Client
from dotenv import load_dotenv
load_dotenv()

# Twilio credentials
account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
auth_token = os.environ.get('TWILIO_AUTH_TOKEN')

client = Client(account_sid, auth_token)

message = client.messages.create(
    to='+919998031139',
    from_='+18129933724',
    body='Hi, this is test sms.'
)

print(message.sid)