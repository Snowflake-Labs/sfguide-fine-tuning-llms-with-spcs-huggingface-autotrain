// Create table of training data
  CREATE OR REPLACE TABLE CONTAINER_HOL_DB.PUBLIC.TRAINING_TABLE AS
  SELECT 
      CONCAT(INSTRUCTION,' ### Metadata: ', METADATA,' ### Response: ', DESCRIPTION) AS "text"
  FROM (
  SELECT
    'You are a customer service assistant with knowledge of product rewards available to customers. Describe this reward offering given reward metadata.' as INSTRUCTION
    ,TO_VARCHAR(OBJECT_CONSTRUCT(*)) AS METADATA
    ,CONCAT
    (
      'Company '
      ,ORGANIZATION_NAME
      ,' is offering a '
      ,REWARD
      ,' '
      ,REWARD_TYPE
      ,' for a '
      ,PRODUCT_CATEGORY
      ,' category product. To receive the reward, '
      ,QUALIFICATION
      ,'. The product description is '
      ,PRODUCT_DESCRIPTION
      ,' More info at '
      ,HYPERLINK
      ,'.'
  ) as DESCRIPTION
  FROM CONTAINER_HOL_DB.PUBLIC.PRODUCT_OFFERS);

  // Create training file in stage
  COPY INTO @CONTAINER_HOL_DB.PUBLIC.VOLUMES/train.csv
  FROM CONTAINER_HOL_DB.PUBLIC.TRAINING_TABLE
  FILE_FORMAT = (
          TYPE = CSV
          FIELD_OPTIONALLY_ENCLOSED_BY = '"'
          ESCAPE_UNENCLOSED_FIELD = NONE
          COMPRESSION = None
      )
    OVERWRITE = TRUE
    SINGLE=TRUE
    HEADER=TRUE;