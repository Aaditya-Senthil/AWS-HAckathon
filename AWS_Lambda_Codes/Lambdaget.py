import os
import json
import boto3
import urllib.request
import urllib.error
import requests
import logging
import traceback
from datetime import datetime, timezone
from botocore.exceptions import ClientError
from decimal import Decimal
import concurrent.futures

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# AWS Setup
DDB_TABLE = os.environ['DDB_TABLE_NAME']
LLAMA_API_KEY = os.environ['LLAMA_CLOUD_API_KEY']
ddb = boto3.resource('dynamodb').Table(DDB_TABLE)
JINA_API_KEY = os.environ['JINA_API_KEY']

def to_decimal_list(float_list):
    return [Decimal(str(x)) for x in float_list]


def get_embedding(text):
    try:
        response = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers={
                "Authorization": f"Bearer {JINA_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "input": [text],
                "model": "jina-embeddings-v2-base-en"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            embedding = response.json()["data"][0]["embedding"]
            return embedding
        else:
            logger.error(f"❌ Jina API error: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"⚠️ Jina embedding failed: {e}")
        return None
                
def generate_embeddings_for_content(parsed_content, extracted_data):
    """Generate embeddings for both parsed content and extracted data"""
    embeddings = {}
    
    if parsed_content:
        logger.info("📝 Generating embedding for parsed content")
        embedding = get_embedding(parsed_content)
        if embedding:
            embeddings['parsed_content_embedding'] = embedding
    
    if extracted_data:
        try:
            if isinstance(extracted_data, dict):
                searchable_text = json.dumps(extracted_data, ensure_ascii=False)
            else:
                searchable_text = str(extracted_data)
            
            logger.info("🔍 Generating embedding for extracted data")
            embedding = get_embedding(searchable_text)
            if embedding:
                embeddings['extracted_data_embedding'] = embedding
                
        except Exception as e:
            logger.error(f"⚠️ Failed to process extracted data for embedding: {e}")
    
    return embeddings

def lambda_handler(event, context):
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type,X-User-Id",
        "Content-Type": "application/json"
    }

    try:
        logger.info(f"Event received: {json.dumps(event, default=str)}")

        # Safe parameter extraction
        path_params = event.get('pathParameters') or {}
        headers_raw = {k.lower(): v for k, v in (event.get('headers') or {}).items()}
        query_params = event.get('queryStringParameters') or {}

        job_id = path_params.get('job_id')
        user_id = headers_raw.get('user-id') or query_params.get('user_id') or headers_raw.get('x-user-id')

        logger.info(f"Extracted - job_id: {job_id}, user_id: {user_id}")

        if not job_id:
            return error_response("Missing job_id in path", headers)

        if not user_id:
            return error_response("Missing user ID", headers)

        # Get job from DynamoDB
        job = get_job_record(job_id, user_id)
        if not job:
            return error_response("Job not found", headers, 404)

        logger.info(f"Current job overall_status: {job.get('overall_status')}")
        logger.info(f"Parse status: {job.get('parse_status')}, Extract status: {job.get('extract_status')}")

        # Return cached result if already completed
        if job['overall_status'] in ['SUCCESS', 'FAILED']:
            logger.info("Returning cached result")
            return format_success_response(job, headers)

        # Fetch results from both APIs in parallel
        logger.info("Fetching parse and extract results in parallel...")
        parse_result, extract_result = fetch_parallel_results(
            job.get('parse_job_id'), 
            job.get('extract_job_id')
        )
        
        logger.info(f"Parse result status: {parse_result.get('status')}")
        logger.info(f"Extract result status: {extract_result.get('status')}")
        
        # Update job based on both API results
        updated_job = update_job_status(job_id, user_id, job, parse_result, extract_result)
        logger.info(f"Updated job overall_status: {updated_job.get('overall_status')}")

        return format_success_response(updated_job, headers)

    except Exception as e:
        logger.error(f"Error in lambda_handler: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return error_response("Internal server error", headers, 500)

def get_job_record(job_id, user_id):
    """Get job record with user validation"""
    try:
        response = ddb.get_item(Key={'job_id': job_id})
        job = response.get('Item')
        
        if not job:
            logger.info(f"No job found for ID: {job_id}")
            return None

        # Validate user owns this job
        if job.get('user_id') != user_id:
            logger.info(f"User mismatch: {job.get('user_id')} vs {user_id}")
            return None

        return job
    except ClientError as e:
        logger.error(f"DynamoDB error: {e.response['Error']['Message']}")
        return None

def fetch_parallel_results(parse_job_id, extract_job_id):
    """Fetch both parse and extract results in parallel"""
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both API calls simultaneously
            parse_future = executor.submit(fetch_parse_results, parse_job_id) if parse_job_id else None
            extract_future = executor.submit(fetch_extract_results, extract_job_id) if extract_job_id else None
            
            # Get results
            parse_result = parse_future.result(timeout=30) if parse_future else {'status': 'FAILED', 'error': 'No parse job ID'}
            extract_result = extract_future.result(timeout=30) if extract_future else {'status': 'FAILED', 'error': 'No extract job ID'}
            
            return parse_result, extract_result
            
    except Exception as e:
        logger.error(f"Error in parallel fetch: {str(e)}")
        return (
            {'status': 'FAILED', 'error': f'Parse fetch failed: {e}'}, 
            {'status': 'FAILED', 'error': f'Extract fetch failed: {e}'}
        )

def fetch_parse_results(parse_job_id):
    """Get parsing result using the correct URL format"""
    if not parse_job_id:
        return {'status': 'FAILED', 'error': 'No parse job ID'}
    
    # Using the URL format you provided
    result_endpoint = f"https://api.cloud.llamaindex.ai/api/parsing/job/{parse_job_id}/result/markdown"
    
    logger.info(f"🔍 Fetching parse results for job: {parse_job_id}")
    logger.info(f"📍 Parse result endpoint: {result_endpoint}")
    
    try:
        req = urllib.request.Request(result_endpoint)
        req.add_header('Authorization', f'Bearer {LLAMA_API_KEY}')
        req.add_header('Accept', 'text/markdown')  # Accept markdown format
        req.add_header('User-Agent', 'AWS-Lambda-Python/1.0')
        
        logger.info(f"🔐 Using API key: {LLAMA_API_KEY[:10]}...{LLAMA_API_KEY[-5:]}")
        
        with urllib.request.urlopen(req, timeout=15) as response:
            if response.status == 200:
                # Parse result is markdown text, not JSON
                markdown_content = response.read().decode('utf-8')
                logger.info(f"✅ Got parse result (length: {len(markdown_content)})")
                
                return {
                    'status': 'SUCCESS',
                    'parsed_content': markdown_content,
                    'content_type': 'markdown'
                }
            else:
                logger.error(f"❌ Parse result fetch failed: {response.status}")
                return {'status': 'PROCESSING'}  # Still processing
                
    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.info("🔍 Parse result not ready (404)")
            return {'status': 'PROCESSING'}
        elif e.code in [401, 403]:
            logger.error(f"🔐 Parse auth error: {e.code} {e.reason}")
            return {'status': 'FAILED', 'error': f'Parse auth error: {e.code}'}
        else:
            logger.error(f"❌ Parse HTTP {e.code}: {e.reason}")
            return {'status': 'FAILED', 'error': f'Parse HTTP error: {e.code}'}
            
    except Exception as e:
        logger.error(f"⚠️ Parse fetch error: {str(e)}")
        return {'status': 'FAILED', 'error': f'Parse fetch failed: {e}'}

def fetch_extract_results(extract_job_id):
    """
    Fetch extraction result with enhanced logic, matching Method 2 style.
    """
    if not extract_job_id:
        return {'status': 'FAILED', 'error': 'No extract job ID provided'}
    
    status_endpoint = f"https://api.cloud.llamaindex.ai/api/v1/extraction/jobs/{extract_job_id}"
    result_endpoint = f"https://api.cloud.llamaindex.ai/api/v1/extraction/jobs/{extract_job_id}/result"

    logger.info(f"🔍 Checking extract job: {extract_job_id}")
    logger.info(f"📍 Status endpoint: {status_endpoint}")
    logger.info(f"📍 Result endpoint: {result_endpoint}")
    logger.info(f"🔐 Using API key: {LLAMA_API_KEY[:10]}...{LLAMA_API_KEY[-5:]}")
    
    # Step 1: Check Job Status
    try:
        req = urllib.request.Request(status_endpoint)
        req.add_header('Authorization', f'Bearer {LLAMA_API_KEY}')
        req.add_header('Accept', 'application/json')
        req.add_header('User-Agent', 'AWS-Lambda-Python/1.0')

        with urllib.request.urlopen(req, timeout=15) as response:
            if response.status == 200:
                status_data = response.read().decode('utf-8')
                status_result = json.loads(status_data)
                logger.info(f"✅ Job status response: {json.dumps(status_result, indent=2)}")
                
                job_status = status_result.get('status', '').upper()
                logger.info(f"🔄 Extract job status: {job_status}")
                
                if job_status == 'SUCCESS':
                    logger.info("🎯 Job completed successfully, proceeding to fetch result.")
                elif job_status in ['PROCESSING', 'PENDING']:
                    logger.info(f"⏳ Job is still {job_status.lower()}")
                    return {'status': job_status}
                elif job_status == 'FAILED':
                    logger.error("❌ Extract job failed on remote service")
                    return {'status': 'FAILED', 'error': 'Job failed on external service'}
                else:
                    logger.warning(f"🔄 Unknown job status: {job_status}")
                    return {'status': 'PENDING'}
            else:
                logger.error(f"❌ Status check failed with HTTP {response.status}")
    except urllib.error.HTTPError as e:
        logger.error(f"❌ Status HTTP Error: {e.code} {e.reason}")
        if e.code == 404:
            return {'status': 'FAILED', 'error': 'Job not found on remote service'}
        elif e.code in [401, 403]:
            return {'status': 'FAILED', 'error': f'Authentication error: {e.code}'}
        else:
            return {'status': 'FAILED', 'error': f'Status check HTTP error: {e.code}'}
    except Exception as e:
        logger.error(f"⚠️ Status check error: {str(e)}")
        return {'status': 'FAILED', 'error': f'Status check failed: {e}'}
    
    # Step 2: Fetch Extraction Result
    try:
        req = urllib.request.Request(result_endpoint)
        req.add_header('Authorization', f'Bearer {LLAMA_API_KEY}')
        req.add_header('Accept', 'application/json')
        req.add_header('User-Agent', 'AWS-Lambda-Python/1.0')

        logger.info(f"📥 Fetching result from: {result_endpoint}")
        
        with urllib.request.urlopen(req, timeout=15) as response:
            if response.status == 200:
                data = response.read().decode('utf-8')
                result = json.loads(data)
                logger.info(f"✅ Got extract result: {json.dumps(result, indent=2)}")

                if is_valid_extraction(result):
                    return {
                        'status': 'SUCCESS',
                        'extracted_data': result
                    }
                else:
                    logger.warning("⚠️ Invalid extraction data")
                    return {'status': 'FAILED', 'error': 'Invalid extraction data'}
            else:
                logger.error(f"❌ Result fetch failed with status: {response.status}")
    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode('utf-8')
        except:
            pass
        logger.error(f"❌ HTTP {e.code} error when fetching result")
        if e.code == 400:
            return {'status': 'FAILED', 'error': f'Bad request: {error_body[:200]}'}
        elif e.code == 404:
            logger.info("🔍 Result not ready (404) - even though job was SUCCESS")
            return {'status': 'PROCESSING'}
        elif e.code in [401, 403]:
            return {'status': 'FAILED', 'error': f'Authentication error: {e.code}'}
        else:
            return {'status': 'FAILED', 'error': f'Result HTTP error: {e.code}'}
    except urllib.error.URLError as e:
        logger.error(f"🌐 Network error: {e.reason}")
        return {'status': 'FAILED', 'error': f'Network error: {e.reason}'}
    except Exception as e:
        logger.error(f"⚠️ Result fetch error: {str(e)}")
        return {'status': 'FAILED', 'error': f'Result fetch failed: {e}'}

    # Final fallback
    return {
        'status': 'FAILED',
        'error': 'Job completed but could not retrieve results'
    }
  
def is_valid_extraction(data):
    """Check if extraction data is valid"""
    if not data:
        return False
    if isinstance(data, dict):
        if not data or 'error' in data or 'message' in data:
            return False
        return True
    return True

def update_job_status(job_id, user_id, current_job, parse_result, extract_result):
    """Update job status in DynamoDB based on both API results"""
    try:
        now = datetime.now(timezone.utc).isoformat()
        
        # Determine individual statuses
        parse_status = parse_result.get('status', 'FAILED')
        extract_status = extract_result.get('status', 'FAILED')
        
        # Determine overall status
        if parse_status == 'SUCCESS' and extract_status == 'SUCCESS':
            overall_status = 'SUCCESS'
        elif parse_status == 'FAILED' or extract_status == 'FAILED':
            overall_status = 'FAILED'
        else:
            overall_status = 'PROCESSING'

        # Prepare update expression
        update_expr = 'SET parse_status = :parse_status, extract_status = :extract_status, overall_status = :overall_status, updated_at = :now'
        expr_values = {
            ':parse_status': parse_status,
            ':extract_status': extract_status,
            ':overall_status': overall_status,
            ':now': now
        }
        expr_names = {}

        # Add parsed content if available
        parsed_content = None
        if parse_status == 'SUCCESS' and 'parsed_content' in parse_result:
            parsed_content = parse_result['parsed_content']
            update_expr += ', parsed_content = :parsed_content'
            expr_values[':parsed_content'] = parsed_content

        # Add extracted data if available
        extracted_data = None
        if extract_status == 'SUCCESS' and 'extracted_data' in extract_result:
            extracted_data = extract_result['extracted_data']
            result_data = json.dumps(extracted_data)
            update_expr += ', extracted_data = :extracted_data'
            expr_values[':extracted_data'] = result_data

        # Generate embeddings if both operations succeeded
        if overall_status == 'SUCCESS' and (parsed_content or extracted_data):
            logger.info("🔮 Generating embeddings for successful job")
            embeddings = generate_embeddings_for_content(parsed_content, extracted_data)
            
            # Add embeddings to update if generated successfully
            if embeddings.get('parsed_content_embedding'):
                update_expr += ', parsed_content_embedding = :parsed_embedding'
                expr_values[':parsed_embedding'] = to_decimal_list(embeddings['parsed_content_embedding'])
                
            if embeddings.get('extracted_data_embedding'):
                update_expr += ', extracted_data_embedding = :extracted_embedding'
                expr_values[':extracted_embedding'] = to_decimal_list(embeddings['extracted_data_embedding'])

        # Add error messages if any
        errors = []
        if parse_status == 'FAILED' and 'error' in parse_result:
            errors.append(f"Parse: {parse_result['error']}")
        if extract_status == 'FAILED' and 'error' in extract_result:
            errors.append(f"Extract: {extract_result['error']}")
        
        if errors:
            update_expr += ', error_message = :error'
            expr_values[':error'] = '; '.join(errors)

        logger.info(f"Update expression: {update_expr}")

        # Update DynamoDB
        response = ddb.update_item(
            Key={'job_id': job_id},
            UpdateExpression=update_expr,
            ExpressionAttributeValues=expr_values,
            ReturnValues='ALL_NEW'
        )

        updated_job = response['Attributes']
        logger.info(f"DynamoDB update successful. Overall status: {updated_job.get('overall_status')}")
        return updated_job

    except ClientError as e:
        logger.error(f"DB update failed: {e.response['Error']['Message']}")
        # Return current job with updated status for response
        current_job.update({
            'parse_status': parse_status,
            'extract_status': extract_status,
            'overall_status': overall_status,
            'updated_at': now
        })
        
        if parse_status == 'SUCCESS' and parsed_content:
            current_job['parsed_content'] = parsed_content
        if extract_status == 'SUCCESS' and extracted_data:
            current_job['extracted_data'] = json.dumps(extracted_data)
            
        return current_job

def format_success_response(job, headers):
    """
    Format success response with both parse and extract data
    Using detailed flow and explicit status handling like Method 2.
    """
    logger.info(f"🔍 Formatting response for job_id: {job.get('job_id')}")
    
    response_data = {
        'success': True,
        'job_id': job.get('job_id'),
        'overall_status': job.get('overall_status', 'PROCESSING'),
        'filename': job.get('filename', ''),
        'message': f"Overall Status: {job.get('overall_status', 'PROCESSING')}"
    }
    
    # Add embedding information if available
    if job.get('parsed_content_embedding'):
        response_data['has_parsed_content_embedding'] = True
        response_data['parsed_content_embedding_size'] = len(job['parsed_content_embedding'])
    
    if job.get('extracted_data_embedding'):
        response_data['has_extracted_data_embedding'] = True
        response_data['extracted_data_embedding_size'] = len(job['extracted_data_embedding'])
    
    # Try to parse extracted data if available
    if job.get('extracted_data'):
        try:
            logger.info("📦 Parsing extracted_data field")
            extracted_data = job['extracted_data']
            if isinstance(extracted_data, str):
                extracted_data = json.loads(extracted_data)
            response_data['data'] = extracted_data
            logger.info("✅ Extracted data parsed successfully")
        except (TypeError, json.JSONDecodeError) as e:
            logger.warning(f"⚠️ Failed to parse extracted_data: {str(e)}")
            response_data['extracted_data'] = job['extracted_data']
    
    # Explicit status handling
    status = job.get('overall_status', 'PROCESSING').upper()
    logger.info(f"🔄 Overall job status: {status}")
    
    if status == 'SUCCESS':
        response_data['message'] = '🎉 Processing completed successfully!'
    elif status == 'FAILED':
        error_msg = job.get('error_message', 'Unknown error')
        response_data['error'] = error_msg
        response_data['message'] = '❌ Processing failed'
        logger.error(f"❌ Job failed with error: {error_msg}")
    elif status in ['PROCESSING', 'PENDING']:
        response_data['message'] = '⏳ Processing in progress...'
    else:
        response_data['message'] = f"ℹ️ Status: {status}"
        logger.info(f"ℹ️ Received unhandled status: {status}")
    
    # Build and return final HTTP response
    final_response = {
        'statusCode': 200,
        'body': json.dumps(response_data, default=str),
        'headers': headers
    }
    
    logger.info(f"📤 Formatted response ready for job_id: {job.get('job_id')}")
    return final_response

def error_response(message, headers, status_code=400):
    """Error response with consistent structure"""
    return {
        'statusCode': status_code,
        'body': json.dumps({
            'success': False,
            'error': message
        }),
        'headers': headers
    }