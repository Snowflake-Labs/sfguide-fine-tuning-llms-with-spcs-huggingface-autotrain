USE ROLE ACCOUNTADMIN;
CREATE SECURITY INTEGRATION IF NOT EXISTS snowservices_ingress_oauth
  TYPE=oauth
  OAUTH_CLIENT=snowservices_ingress
  ENABLED=true;

CREATE OR REPLACE NETWORK RULE ALLOW_ALL_RULE
  TYPE = 'HOST_PORT'
  MODE = 'EGRESS'
  VALUE_LIST= ('0.0.0.0:443', '0.0.0.0:80');

CREATE EXTERNAL ACCESS INTEGRATION ALLOW_ALL_EAI
  ALLOWED_NETWORK_RULES = (ALLOW_ALL_RULE)
  ENABLED = true;

GRANT USAGE ON INTEGRATION ALLOW_ALL_EAI TO ROLE CONTAINER_USER_ROLE;

USE ROLE CONTAINER_USER_ROLE;
USE DATABASE CONTAINER_HOL_DB;

CREATE IMAGE REPOSITORY CONTAINER_HOL_DB.PUBLIC.IMAGE_REPO;

CREATE COMPUTE POOL IF NOT EXISTS CONTAINER_HOL_POOL_GPU_NV_M
MIN_NODES = 1
MAX_NODES = 1
INSTANCE_FAMILY = GPU_NV_M
AUTO_RESUME = true;