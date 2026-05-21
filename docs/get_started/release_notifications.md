# Release Notifications

Get notified when new {{ dlc_short }} images are published. Notifications include version, metadata, and regional image URIs.

## SNS Topic

```
arn:aws:sns:us-west-2:767397762724:dlc-updates
```

## Subscribe via CLI

```bash
aws sns subscribe \
  --topic-arn arn:aws:sns:us-west-2:767397762724:dlc-updates \
  --protocol email \
  --notification-endpoint your-email@example.com \
  --region us-west-2
```

Check your email and confirm the subscription.

## Subscribe via Console

1. Open the [SNS console](https://console.aws.amazon.com/sns/home?region=us-west-2#/subscriptions){:target="_blank"} (US West - Oregon)
2. Choose **Create subscription**
3. Set **Topic ARN** to `arn:aws:sns:us-west-2:767397762724:dlc-updates`
4. Set **Protocol** to Email (or SQS, Lambda, etc.)
5. Enter your endpoint and choose **Create subscription**
6. Confirm via the email you receive
