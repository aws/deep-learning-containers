## Agent Workload using hosted Amazon Bedrock Model
Running Agent workload involve using SeeAct framework and browswer automation to start interactive session with the AI Agent and perform the browser operations. Therefore, this part is recommended to be performed on local machine for browswer access.

### Clone SeeAct Repository
First clone the customized SeeAct repository that contains example code that can work with Amazon Bedrock, as well as a couple of test scripts.

```
git clone https://github.com/junpuf/SeeAct.git
git checkout sglang
```

### setup SeeAct guide to setup local runtime environment
Follow the setup guide [here](https://github.com/junpuf/SeeAct/tree/sglang?tab=readme-ov-file#seeact-tool)

### Test Playwright -- browswer automation tool used by SeeAct
```
cd SeeAct/src
python test_playwright.py
```

### Test Bedrock Model availability 
Modify the content of `test_bedrock.py`. Update the `MODEL_ID` to be your hosted Amazon Bedrock model ARN, as well as setting AWS connection.
```
export AWS_ACCESS_KEY_ID="replace with your aws credential"
export AWS_SECRET_ACCESS_KEY="replace with your aws credential"
export AWS_SESSION_TOKEN="replace with your aws credential"
```
Run the test
```
pytohn test_bedrock.py
```
> Note： If `botocore.errorfactory.ModelNotReadyException` error occurs, please retry the command in a few minutes.

### Run the agent workflow
The branch has already added support for BedrockEngine and SGLang for running inference with finetuned llama 3.2 vision model. The default option use Bedrock inference.

To run demo, again update the `self.model` from `src/demo_utils/inference_engine.py at line 229` to your Amazon Bedrock Model ARN. Then run the following.
```
python seeact.py -c config/demo_mode.toml 
```

```markdown
![Agent workflow screenshot showing browser automation](./agent_screenshot_1.png)
```
