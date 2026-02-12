# Receive Notifications on New Updates

You can receive notifications whenever a new {{ dlc_short }} is released. Notifications are published with
[Amazon Simple Notification Service ({{ sns }})](https://docs.aws.amazon.com/sns/latest/dg/welcome.html){:target="_blank"} using the following topic.

```
arn:aws:sns:us-west-2:767397762724:dlc-updates
```

Messages are posted here when a new {{ dlc_short }} is published. The version, metadata, and regional image URIs of the container will be included in
the message.

These messages can be received using several different methods. We recommend the following method.

## Subscribe to the SNS Topic

1. Open the [{{ sns }} console](https://console.aws.amazon.com/sns/home){:target="_blank"}.

2. In the navigation bar, change the {{ aws }} Region to **US West (Oregon)**, if necessary. You must select the region where the {{ sns }}
   notification that you are subscribing to was created.

3. In the navigation pane, choose **Subscriptions**, **Create subscription**.

4. For the **Create subscription** dialog box, do the following:

   1. For **Topic ARN**, copy and paste the following Amazon Resource Name (ARN): `arn:aws:sns:us-west-2:767397762724:dlc-updates`

   2. For **Protocol**, choose one from **[Amazon Simple Queue Service ({{ sqs }}), {{ lambda }}, Email, Email-JSON]**

   3. For **Endpoint**, enter the email address or **Amazon Resource Name (ARN)** of the resource that you will use to receive the notifications.

   4. Choose **Create subscription**.

5. You receive a confirmation email with the subject line *{{ aws }} Notification - Subscription Confirmation*. Open the email and choose **Confirm
   subscription** to complete your subscription.
