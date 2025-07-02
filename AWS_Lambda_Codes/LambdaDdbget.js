import { DynamoDBClient } from "@aws-sdk/client-dynamodb";
import { DynamoDBDocumentClient, ScanCommand } from "@aws-sdk/lib-dynamodb";

const client = new DynamoDBClient({});
const dynamodb = DynamoDBDocumentClient.from(client);

export const handler = async (event) => {
    const headers = {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token',
        'Access-Control-Allow-Methods': 'GET,OPTIONS'
    };

    try {
        console.log('Event received:', JSON.stringify(event, null, 2));
        
        const user_id = event.queryStringParameters?.user_id || event.pathParameters?.user_id;
        
        if (!user_id) {
            return {
                statusCode: 400,
                headers,
                body: JSON.stringify({
                    error: 'user_id is required',
                    debug: {
                        queryStringParameters: event.queryStringParameters,
                        pathParameters: event.pathParameters
                    }
                })
            };
        }

        const params = {
            TableName: 'LlamaDocument_Extarction',
            FilterExpression: 'user_id = :user_id',
            ExpressionAttributeValues: { ':user_id': user_id },
            ProjectionExpression: 'job_id, filename, #status, created_at, updated_at, config, s3_key, #result',
            ExpressionAttributeNames: { '#status': 'extract_status',  '#result': 'extracted_data' }
        };

        const result = await dynamodb.send(new ScanCommand(params));

        const formattedData = result.Items.map(item => ({
            id: item.job_id,
            fileName: item.filename,
            jobId: item.job_id,
            status: item.extract_status,
            createdAt: formatDate(item.created_at),
            completedAt: formatDate(item.updated_at),
            config: tryParseJSON(item.config),
            data: tryParseJSON(item.extracted_data),
            documentUrl: item.s3_key ? `https://your-bucket.s3.amazonaws.com/${item.s3_key}` : null,
        }));

        formattedData.sort((a, b) => 
            new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
        );

        return {
            statusCode: 200,
            headers,
            body: JSON.stringify({
                success: true,
                data: formattedData,
                count: formattedData.length
            })
        };

    } catch (error) {
        console.error('Error:', error);
        return {
            statusCode: 500,
            headers,
            body: JSON.stringify({
                error: 'Internal server error',
                message: error.message
            })
        };
    }
};

function formatDate(dateString) {
    if (!dateString) return null;
    try {
        const date = new Date(dateString);
        return isNaN(date) ? dateString : date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    } catch {
        return dateString;
    }
}

function tryParseJSON(str) {
    try {
        return JSON.parse(str || '{}');
    } catch {
        return {};
    }
}
