import boto3

# Initialize SNS client
sns = boto3.client("sns", region_name="ap-south-1")  # Use the region that supports SMS

# Replace with your mobile number in E.164 format
phone_number = "+919998031139"

# Publish SMS message
response = sns.publish(
    PhoneNumber=phone_number,
    Message="Default SMS",
    MessageAttributes={
        'AWS.SNS.SMS.SMSType': {
            'DataType': 'String',
            'StringValue': 'Transactional'  # Or 'Promotional'
        }
    }
)

print("Message sent! Message ID:", response["MessageId"])
