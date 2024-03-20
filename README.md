# LLM Fine-Tuning with AutoTrain in Snowflake
Created by Jason Summer

![Hugging Face and Snowflake](images/logos.png? "Hugging Face and Snowflake") 


Snowflake announced Snowpark Container Services (SPCS) in 2023, enabling virtually any workload to run securely in Snowflake. SPCS also provides GPU compute, which is critical for those tasked with GenAI use cases. In this brief tutorial, I will illustrate how to run Hugging Face’s AutoTrain Advanced entirely in a Snowpark container with GPU compute to fine-tune open-sourced LLMs using data in Snowflake.

## Case for Fine-Tuning
Out of the box, LLMs can comprehend natural language but tend to fall short in use cases that require additional context, behavior, or tone. While context or specific instructions can be dynamically passed to the LLM, the amount of context, knowledge, or instruction may exceed the context window of a given LLM. In such cases, fine-tuning, or training a model on input-output pairs can be considered. As a simplified example, we will be fine-tuning an LLM to understand product metadata in its raw form.

## Snowpark Container Services Primer
SPCS is a fully managed container offering designed to facilitate the deployment, management, and scaling of containerized applications within the Snowflake ecosystem. Visit Snowflake [SPCS documentation](https://docs.snowflake.com/developer-guide/snowpark-container-services/overview) for more details and [tutorials](https://docs.snowflake.com/developer-guide/snowpark-container-services/overview-tutorials). These tutorials will provide a baseline understanding for subsequent steps.

SPCS simplifies the architectural considerations of containerized services, and enables us to bring the LLM to the data instead of extracting data out of Snowflake. This provides a much more secure framework.

## Advanced AutoTrain
Hugging Face advertises [AutoTrain](https://huggingface.co/docs/autotrain/index) as a no code, hosted service to train state-of-the-art models for any task. However, reading further reveals AutoTrain Advanced, which can be pip-installed to enable fine-tuning locally. In our case, this means the training will occur inside of SPCS.

## Setup
To start, create a database, image repository, two stages (for service specs & training data), and a GPU compute pool in Snowsight.
```
CREATE DATABASE IF NOT EXISTS AUTOTRAIN_DB;
 
USE DATABASE AUTOTRAIN_DB;
 
CREATE IMAGE REPOSITORY AUTOTRAIN_DB.PUBLIC.IMAGES;
 
-- Stage to store the service spec file
CREATE OR REPLACE STAGE SPEC_STAGE;
-- Stage to store training csv file
CREATE OR REPLACE STAGE DATA_STAGE ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');
 
CREATE COMPUTE POOL IF NOT EXISTS GPU_NV_M_CP
MIN_NODES = 1
MAX_NODES = 1
INSTANCE_FAMILY = GPU_NV_M
AUTO_RESUME = true;
```

2. Next, data will be placed in our data stage. At the time of this tutorial, AutoTrain requires a csv file named `train.csv` with a single column named `text` for LLM [Supervised Fine-Tuning (SFT)](https://huggingface.co/docs/autotrain/llm_finetuning).

It’s important for training records to mirror the underlying model’s original prompt format. As an example, I want a fine-tuned llama-2 model to understand product reward metadata in JSON format.

```
SELECT text FROM AUTOTRAIN_TBL SAMPLE (1 ROWS);

-- query result:
<s>[INST] <<SYS>>
You are a customer service assistant with knowledge of product rewards available to customers. Describe this reward offering given reward metadata.
<</SYS>>
{"HYPERLINK":"www.gamingenthusiasts.com/gamingheadset","ORGANIZATION_NAME":"Gaming Enthusiasts","PRODUCT_CATEGORY":"Gaming","PRODUCT_DESCRIPTION":"Immerse yourself in the gaming world with our Gaming Headset. This high-quality headset offers crystal-clear audio, noise cancellation, and a comfortable fit, enhancing your gaming experience. Level up your gaming setup with Gaming Enthusiasts' Gaming Headset.","PRODUCT_NAME":"Gaming Headset","QUALIFICATION":"Purchase Gaming Headset and enter code GAMER20 at checkout","REWARD":"$20","REWARD_TYPE":"Code"} 
[/INST] Company Gaming Enthusiasts is offering a $20 discount code for a Gaming category product. To receive the reward, Purchase Gaming Headset and enter code GAMER20 at checkout. The product description is Immerse yourself in the gaming world with our Gaming Headset. This high-quality headset offers crystal-clear audio, noise cancellation, and a comfortable fit, enhancing your gaming experience. Level up your gaming setup with Gaming Enthusiasts' Gaming Headset. More info at www.gamingenthusiasts.com/gamingheadset. </s>
```

The csv file, `train.csv`, for fine-tuning can be created as shown below in Snowsight.

```
COPY INTO @AUTOTRAIN_DB.PUBLIC.DATA_STAGE/train.csv
FROM AUTOTRAIN_TBL
FILE_FORMAT = (
         TYPE = CSV
         FIELD_OPTIONALLY_ENCLOSED_BY = '"'
         ESCAPE_UNENCLOSED_FIELD = NONE
         COMPRESSION = None
    )
  OVERWRITE = TRUE
  SINGLE=TRUE
  HEADER=TRUE;
```

3. Next, craft the service spec file. We will continue with llama-2–7b in this example. Set GPU requests and limits to 2 to compare the base and fine-tuned model in a single chat interface in the final step. Otherwise, set them to 1 to only chat with the fine-tuned model.

Two endpoints are included for demonstration purposes: `jupyter` and `app`. Jupyter Lab will be used to check the fine-tuning status while the other endpoint will enable a quick chat-based test of the model. Put the below spec file into `SPEC_STAGE` stage after making updates.

Run `SHOW IMAGE REPOSITORIES` in Snowflake to obtain your Snowflake `REPOSITORY_URL`.

**autotrain.yaml**
```
spec:
  containers:
  - name: autotrain
    image: <YOUR_REPOSITORY_URL>/autotrain
    env:
      SNOWFLAKE_MOUNTED_STAGE_PATH: stage
      MODEL_CARD: meta-llama/Llama-2-7b-hf # Hugging Face model card
      HF_TOKEN: <HF_TOKEN>
      PROJECT_NAME: stage/llama-2-ft # Should be in stage/. Name must mirror model card for prompting
    resources:
      requests:
        nvidia.com/gpu: 2 # <2 for base vs. fine-tuned model, otherwise 1>
      limits:
        nvidia.com/gpu: 2 # <2 for base vs. fine-tuned model, otherwise 1>
    volumeMounts:
    - name: stage
      mountPath: /workspace/stage
  endpoints:
  - name: jupyter # View fine-tuning logs in jupyter terminal
    port: 8888
    public: true
  - name: app # Quick gradio chat app to confirm fine-tuning
    port: 8000
    public: true
  volumes:
  - name: stage
    source: '@DATA_STAGE'
    uid: 1000
    gid: 1000
```

4. Now, create the Dockerfile and corresponding entrypoint.

**autotrain.Dockerfile**

```
FROM nvcr.io/nvidia/rapidsai/rapidsai:23.06-cuda11.8-runtime-ubuntu22.04-py3.10

# Set the working directory
WORKDIR /workspace/

RUN mkdir /workspace/.local /workspace/.cache && chmod 777 -R /workspace
COPY --chmod=777 ./ ./

# Install the dependencies
RUN apt-get update && apt-get install git-lfs && git lfs install
RUN pip install --no-cache-dir --upgrade pip && pip install autotrain-advanced==0.6.79 --force-reinstall && pip install requests botocore torch torchvision torchaudio
RUN apt-get install -y ffmpeg libsm6 libxext6
RUN autotrain setup

# Run Jupyter Notebook on container startup
ENTRYPOINT ["./entrypoint.sh"]
```

**entrypoint.sh**

```
#!/bin/bash
nohup jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password='' > jupyter.out 2>jupyter.err &
echo "Fine-funing starting..."
nohup autotrain llm --train --project-name $PROJECT_NAME --model $MODEL_CARD --data-path stage/ --text_column text --use-peft --quantization int4 --lr 1e-4 --train-batch-size 4 --epochs 3 --trainer sft --token $HF_TOKEN --merge_adapter > autotrain.out 2>autotrain.err &
( tail -f -n0 autotrain.err & ) | grep -q "Saving target model..."
echo "Fine-funing complete. Merged model saved to stage."
tail -f /dev/null # Keeps container running
```

The above entrypoint will launch Jupyter Lab and commence fine-tuning at start up.

Build and push the above image to SPCS.

```
docker build --rm --platform linux/amd64 -f autotrain.Dockerfile -t <YOUR_REPOSITORY_URL>/autotrain .
docker push <YOUR_REPOSITORY_URL>/autotrain
```
5. We can now start the container service in Snowsight.

```
CREATE SERVICE AUTOTRAIN
  IN COMPUTE POOL GPU_NV_M_CP
  FROM @SPEC_STAGE
  SPECIFICATION_FILE = autotrain.yaml
  MIN_INSTANCES = 1
  MAX_INSTANCES = 1
  COMMENT = '{"origin": "sf_sit",
             "name": "hf_autotrain",
             "version": {"major": 1, "minor": 1},
             "attributes":{"solution_family":"spcs"}}';
```

Wait for the service status to confirm it is `RUNNING` with system function `GET_SERVICE_STATUS`. Once confirmed, visit the `jupyter` endpoint URL to monitor fine-tuning progress, which is written to `workspace/autotrain.err` in the container working directory. Fine-tuning will likely take some time and is dependent upon chosen model, compute pool, and data. The llama-7b model trained for 56 minutes using a dataset of 1,252 records. A final confirmation will be written to the container logs, accessible via system function `GET_SERVICE_LOGS`.

```
-- Check status and wait for RUNNING
CALL SYSTEM$GET_SERVICE_STATUS('AUTOTRAIN');

-- Obtain endpoint URLs including jupyter
SHOW ENDPOINTS IN SERVICE AUTOTRAIN;

-- Returns "Fine-funing complete. Merged model saved to stage." once fine-tuning complete
CALL SYSTEM$GET_SERVICE_LOGS('AUTOTRAIN', '0', 'autotrain');
```

At this point, the fine-tuning is done and the merged (base + adapter) fine-tuned model resides in Snowflake stage. Below we will run a quick test before suspending the service.

## Quick Test
A Gradio chat interface can be started using the fine-tuned model from Jupyter. Open terminals in Jupyter and run the below commands in order. Wait for each command to complete before running the next. Note that you may need to install `pip install huggingface_hub` and login to hugging face to load the base model for comparison. 

```
pip3 install "fschat[model_worker,webui]"

python3 -m fastchat.serve.controller

# Wait for controller to finish. Open new terminal to run.
CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.model_worker --model-path $PROJECT_NAME --controller http://localhost:21001 --port 31000 --worker http://localhost:31000

# OPTIONAL: Run to include base model in chat for comparison
# Wait for controller to finish. Open new terminal to run.
# You may need to pip install huggingface_hub and login to HuggingFace first with huggingface-cli login --token $HF_TOKEN
CUDA_VISIBLE_DEVICES=1 python3 -m fastchat.serve.model_worker --model-path $MODEL_CARD --controller http://localhost:21001 --port 31001 --worker http://localhost:31001

# Wait for model worker(s) to finish. Open new terminal to run.
python3 -m fastchat.serve.gradio_web_server --port 8000
```

Return to Snowsight to obtain the URL for the `app` endpoint (located at port 8000). After using your Snowflake credentials, you will see the Gradio chat interface and evaluate responses. Below you will see a quick comparison of the base model's response with that of the fine-tuned model using a novel offer in the expected JSON structure. Example generations using the base model (llama-2-7b) were mixed oftentimes either getting stuck in a repetitive phrasing or committing verbose hallucations. In comparison, the fine-tuned model reports the desired structure.

<figure>
  <img src=images/comparative_generation.gif  alt="Base model vs. Fine-tuned model">
  <figcaption><i>Prompting base and fine-tuned LLM to describe novel offer using expected JSON structure</i></figcaption>
</figure>

## Final Remarks
Moving forward, fine-tuning can be run as a [job](https://docs.snowflake.com/developer-guide/snowpark-container-services/working-with-jobs) and scheduled in a task. Fine-tuned models can remain in stage or saved to Snowflake model registry.

Lastly, it is worth mentioning that the autotrain-advanced package is under very active development and the API frequently changes.
