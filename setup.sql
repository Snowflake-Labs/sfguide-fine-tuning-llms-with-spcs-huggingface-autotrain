/*
###################
##### SETUP #######
###################
 */
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

/*
###################
### DATA PREP #####
###################
 */
CREATE OR REPLACE TABLE AUTOTRAIN_DB.PUBLIC.AUTOTRAIN_TBL AS 
SELECT 
    CONCAT('<s>[INST] <<SYS>> \n',
            $system_prompt,
            '\n',
            '<</SYS>>',
            '\n\n',
            to_varchar(OBJECT_CONSTRUCT(p.*)),
            ' [/INST] ',
            DESCRIPTION,
            ' </s>') "text"
FROM JSUMMER.CATALOG.PRODUCTS p 
JOIN JSUMMER.CATALOG.DESCRIBER_RESULTS d using (ORGANIZATION_NAME, PRODUCT_NAME, PRODUCT_CATEGORY, HYPERLINK, QUALIFICATION)
;

-- Drop csv of training data into soon to be mounted stage
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

/*
###################
# CREATE SERVICE ##
###################
 */
CREATE SERVICE AUTOTRAIN
IN COMPUTE POOL GPU_NV_M_CP
FROM @SPEC_STAGE
SPECIFICATION_FILE = autotrain.yaml
MIN_INSTANCES = 1
MAX_INSTANCES = 1;

-- Check status and wait for RUNNING
CALL SYSTEM$GET_SERVICE_STATUS('AUTOTRAIN');

-- Obtain endpoint URLs including jupyter
SHOW ENDPOINTS IN SERVICE AUTOTRAIN;

-- Returns "Fine-funing complete. Merged model saved to stage." once fine-tuning complete
CALL SYSTEM$GET_SERVICE_LOGS('AUTOTRAIN', '0', 'autotrain');