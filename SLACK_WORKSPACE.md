# Slack Workspace
For Slack Notification Alert System:

## Option 1: Create your own personal Slack workspace
Instructions:

1.	To send notification alerts to a personal slack workspace, build or update an existing app in your slack workspace to accept incoming webhooks.

2. Copy the webhook url for the app, set up a channel, a bot to recieve the webhook alerts and notifications.

3. Feed the webhook url to the application, which can be done in 2 ways.
   1. [___Recommended___] Export webhook as an environment variable in your system named '**ACTIVENET_WEBHOOK**' and the application will take care of the rest. To export the webhook as an environemnet variable:
      1. On Linux or MacOS, run the command<br>
	  ```export ACTIVENET_WEBHOOK="<insert webhook url here>"```
	  2. On Windows,
         1. Go to Control Panel.
         2. Click on Advanced system settings.
         3. Click Environment Variables. In the section System Variables, click New and set the variable name as 'ACTIVENET_WEBHOOK'.
         4. In the New System Variable window, paste the webhook url.
         5. Close all remaining windows by clicking OK.
   2. Directly paste the webhook url at [demo.py](https://github.com/aaditagarwal/ActiveNet/blob/260cad3e2e34eaf47be842d258de0c3e179b1cb0/demo.py#L104) as<br> ```webhook="<insert webhook url here>"```

## OPTION 2: Use the demo ___active-networkspace___ to receive notification alerts
Instructions:

1. Send us a mail at agarwal.aadit99@gmail.com or aitikgupta@gmail.com with the subject "Slack Demo Workspace invite", and we will issue an invitation for you to join the workspace, as well as the webhook url.
2. Follow instruction 3.2 mentioned above to send and receive notification alerts using the webhook url.

### Note: The webhook url for demo workspace hasn't been mentioned in repository for security reasons.<br>Do not share any webhook url publicly.