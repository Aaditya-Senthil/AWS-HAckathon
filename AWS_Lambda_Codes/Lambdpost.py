import os
import json
import base64
import boto3
import urllib3
from datetime import datetime
import logging
import uuid
import concurrent.futures
import threading

logger = logging.getLogger()
logger.setLevel(logging.INFO)

S3_BUCKET = os.environ.get('S3_BUCKET_NAME', 'Zenathon')
DDB_TABLE = os.environ.get('DDB_TABLE_NAME')
LLAMA_API_KEY = os.environ.get('LLAMA_CLOUD_API_KEY')
EXTRACTION_AGENT_ID = os.environ.get('EXTRACTION_AGENT_ID', '7949ce63-fbd0-41cc-9264-4e427658eb25')

s3_client = boto3.client('s3')
ddb_client = boto3.client('dynamodb')

http = urllib3.PoolManager()

def lambda_handler(event, context):
    try:
        if event.get('httpMethod') == 'OPTIONS':
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Methods': 'POST, OPTIONS',
                    'Access-Control-Allow-Headers': 'Content-Type, Authorization'
                },
                'body': ''
            }
        
        if 'Records' in event:
            return handle_sqs_messages(event, context)
        else:
            return handle_http_request(event, context)
    
    except Exception as e:
        logger.error(f"Unhandled error in lambda_handler: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization'
            },
            'body': json.dumps({
                'error': 'Internal server error',
                'message': str(e)
            })
        }

def handle_http_request(event, context):
    try:
        if 'body' not in event:
            raise ValueError("No request body found")
        
        request_body = json.loads(event['body'])
        filename = request_body.get('filename')
        file_content = request_body.get('file_content')  
        user_id = request_body.get('user_id')
        config = request_body.get('config', {})
        
        if not all([filename, file_content, user_id]):
            raise ValueError("Missing required fields: filename, file_content, user_id")
        
        logger.info(f"Processing HTTP request for user {user_id}")
        
        try:
            file_bytes = base64.b64decode(file_content)
        except Exception as e:
            raise ValueError(f"Invalid base64 file content: {e}")
        
        s3_key = upload_to_s3(user_id, filename, file_bytes)
        
        job_id = str(uuid.uuid4())
        
        parse_job_id, extract_job_id = start_parallel_jobs(filename, file_bytes, config)
        
        create_job_record(job_id, user_id, filename, s3_key, parse_job_id, extract_job_id)
        
        logger.info(f"Successfully processed HTTP request for job {job_id}")
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization'
            },
            'body': json.dumps({
                'job_id': job_id,
                'parse_job_id': parse_job_id,
                'extract_job_id': extract_job_id,
                's3_key': s3_key,
                'status': 'PROCESSING'
            })
        }
        
    except Exception as e:
        logger.error(f"Error in handle_http_request: {str(e)}", exc_info=True)
        return {
            'statusCode': 400,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type, Authorization'
            },
            'body': json.dumps({
                'error': 'Bad Request',
                'message': str(e)
            })
        }

def start_parallel_jobs(filename, file_bytes, config):
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            parse_future = executor.submit(start_parse_job, filename, file_bytes, config)
            extract_future = executor.submit(start_extraction_job, filename, file_bytes, config)
            
            parse_job_id = parse_future.result(timeout=30)
            extract_job_id = extract_future.result(timeout=30)
            
            return parse_job_id, extract_job_id
            
    except Exception as e:
        logger.error(f"Error in parallel job execution: {str(e)}")
        raise ValueError(f"Failed to start parallel jobs: {e}")

def start_parse_job(filename, file_bytes, config):
    try:
        fields = {}
        
        if config.get('parsing_instruction'):
            fields['parsing_instruction'] = config['parsing_instruction']
        if config.get('result_type'):
            fields['result_type'] = config['result_type']
        else:
            fields['result_type'] = 'text'
            
        files = {'file': (filename, file_bytes, 'application/pdf')}
        body, content_type = encode_multipart_formdata(fields, files)
        
        response = http.request(
            'POST',
            'https://api.cloud.llamaindex.ai/api/parsing/upload',
            body=body,
            headers={
                'Authorization': f'Bearer {LLAMA_API_KEY}',
                'Content-Type': content_type,
                'Accept': 'application/json'
            },
            timeout=30
        )
        
        if response.status != 200:
            raise Exception(f"Parse API error {response.status}: {response.data.decode()}")
        
        job_data = json.loads(response.data.decode())
        parse_job_id = job_data.get('id')
        
        if not parse_job_id:
            raise ValueError("No parse job ID returned")
            
        logger.info(f"Started parse job: {parse_job_id}")
        return parse_job_id
        
    except Exception as e:
        logger.error(f"Failed to start parse job: {str(e)}")
        raise ValueError(f"Failed to start parsing: {e}")

def start_extraction_job(filename, file_bytes, config):
    try:
        fields = {'extraction_agent_id': EXTRACTION_AGENT_ID}
        
        if config.get('data_schema'):
            fields['data_schema_override'] = json.dumps(config['data_schema'])
        if config.get('extraction_mode'):
            fields['extraction_mode'] = config['extraction_mode']
        
        files = {'file': (filename, file_bytes, 'application/pdf')}
        body, content_type = encode_multipart_formdata(fields, files)
        
        response = http.request(
            'POST',
            'https://api.cloud.llamaindex.ai/api/v1/extraction/jobs/file',
            body=body,
            headers={
                'Authorization': f'Bearer {LLAMA_API_KEY}',
                'Content-Type': content_type,
                'Accept': 'application/json'
            },
            timeout=30
        )
        
        if response.status != 200:
            raise Exception(f"Extract API error {response.status}: {response.data.decode()}")
        
        job_data = json.loads(response.data.decode())
        extraction_job_id = job_data.get('id')
        
        if not extraction_job_id:
            raise ValueError("No extraction job ID returned")
        
        logger.info(f"Started extraction job: {extraction_job_id}")
        return extraction_job_id
        
    except Exception as e:
        logger.error(f"Failed to start extraction job: {str(e)}")
        raise ValueError(f"Failed to start extraction: {e}")

def create_job_record(job_id, user_id, filename, s3_key, parse_job_id, extract_job_id):
    try:
        now = datetime.utcnow().isoformat()
        
        ddb_client.put_item(
            TableName=DDB_TABLE,
            Item={
                'job_id': {'S': job_id},
                'user_id': {'S': user_id},
                'filename': {'S': filename},
                's3_key': {'S': s3_key},
                'parse_job_id': {'S': parse_job_id},
                'extract_job_id': {'S': extract_job_id},
                'parse_status': {'S': 'PROCESSING'},
                'extract_status': {'S': 'PROCESSING'},
                'overall_status': {'S': 'PROCESSING'},
                'created_at': {'S': now},
                'updated_at': {'S': now}
            }
        )
        
        logger.info(f"Created comprehensive job record for {job_id}")
        
    except Exception as e:
        raise ValueError(f"Failed to create job record: {e}")

def handle_sqs_messages(event, context):
    for record in event['Records']:
        try:
            message_body = json.loads(record['body'])
            job_id = message_body['job_id']
            
            logger.info(f"Processing SQS job {job_id}")
            
            update_job_status(job_id, "SQS_PROCESSING")
            
            logger.info(f"Successfully processed SQS job {job_id}")
            
        except Exception as e:
            logger.error(f"Error processing SQS job: {str(e)}", exc_info=True)
    
    return {'statusCode': 200}

def upload_to_s3(user_id, filename, file_bytes):
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    s3_key = f"users/{user_id}/documents/{timestamp}_{filename}"
    
    return s3_key

def update_job_status(job_id, status, error_message=None):
    try:
        now = datetime.utcnow().isoformat()
        
        update_expression = "SET overall_status = :status, updated_at = :updated_at"
        expression_values = {
            ":status": {"S": status},
            ":updated_at": {"S": now}
        }
        
        if error_message:
            update_expression += ", error_message = :error"
            expression_values[":error"] = {"S": error_message}
        
        ddb_client.update_item(
            TableName=DDB_TABLE,
            Key={"job_id": {"S": job_id}},
            UpdateExpression=update_expression,
            ExpressionAttributeValues=expression_values
        )
        
        logger.info(f"Updated job {job_id} status to {status}")
        
    except Exception as e:
        logger.error(f"Failed to update job status: {e}")

def encode_multipart_formdata(fields, files):
    boundary = uuid.uuid4().hex
    body = b''
    
    for key, value in fields.items():
        body += f'--{boundary}\r\n'.encode()
        body += f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode()
        body += f'{value}\r\n'.encode()
    
    for key, (filename, file_data, content_type) in files.items():
        body += f'--{boundary}\r\n'.encode()
        body += f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'.encode()
        body += f'Content-Type: {content_type}\r\n\r\n'.encode()
        body += file_data
        body += b'\r\n'
    
    body += f'--{boundary}--\r\n'.encode()
    content_type = f'multipart/form-data; boundary={boundary}'
    
    return body, content_type
