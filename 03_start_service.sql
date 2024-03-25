USE ROLE CONTAINER_USER_ROLE;
USE DATABASE CONTAINER_HOL_DB;
USE SCHEMA PUBLIC;
CREATE SERVICE AUTOTRAIN
  IN COMPUTE POOL CONTAINER_HOL_POOL_GPU_NV_M
  FROM @SPECS
  SPECIFICATION_FILE = autotrain.yaml
  MIN_INSTANCES = 1
  MAX_INSTANCES = 1
  EXTERNAL_ACCESS_INTEGRATIONS = (ALLOW_ALL_EAI);
  COMMENT = '{"origin": "sf_sit",
             "name": "hf_autotrain",
             "version": {"major": 1, "minor": 1},
             "attributes":{"solution_family":"spcs"}}';

-- Check status and wait for RUNNING
CALL SYSTEM$GET_SERVICE_STATUS('AUTOTRAIN');

-- Obtain endpoint URLs including jupyter
SHOW ENDPOINTS IN SERVICE AUTOTRAIN;

-- Returns "Fine-tuning complete. Merged model saved to stage." once fine-tuning complete
CALL SYSTEM$GET_SERVICE_LOGS('AUTOTRAIN', '0', 'autotrain');