import json
import logging
import jsonschema
import os

from dataclasses import dataclass
from typing import Any, Dict, List

from inference_engine.florence_engine import inferencing

import azure.functions as func
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient


RECORD_ID_TAG = "recordId"
VALUES_TAG = "values"
DATA_TAG = "data"
URL_TAG = "url"
FRAMES_TAG = "frames"
CONTENT_TYPE_TAG = 'Content-Type'
SUBSCRIPTION_ID_TAG = 'apim-subscription-id'
SUBSCRIPTION_ID = 'indexer'
CONTENT_TYPE = 'application/json'
CONTAINER_NAME = 'videos'
CONNECTION_STRING = 'BLOB_CONNECTION_STRING'
FRAME_COUNT_TAG = "FRAME_COUNT"


def extract_video_embeddings(req: func.HttpRequest) -> func.HttpResponse:
    try:
        response_body = handle_request(req)
        return format_http_response(response_body)
    except Exception as e:
        logging.error(f"Invalid request: {req.get_json()}  Error: {e}")
        return func.HttpResponse(f"Invalid request: {req.get_json()}  Error: {e}", status_code=400)


def handle_request(request: func.HttpRequest) -> func.HttpResponse:
    logging.info('RctvVideoIndexer function processed a request.')
    blob_sas_url = os.getenv("BLOB_SAS_URL") or ""
    florence_endpoint = os.getenv("FLORENCE_ENDPOINT") or ""

    request_json = request.get_json()
    logging.info(
        f'RctvVideoIndexer function processed a request: {request_json} .')
    jsonschema.validate(request_json, schema=get_request_schema())

    return handle_request_values(request_json[VALUES_TAG], blob_sas_url, florence_endpoint)


def handle_request_values(values: Any, blob_sas_url: str, florence_endpoint) -> Dict[str, List]:
    response_records = handle_record_entries(
        values, blob_sas_url, florence_endpoint)
    logging.info(f" response records: {response_records}")
    return {VALUES_TAG: response_records}


def handle_record_entries(record_list: List[Dict], blob_sas_url: str, florence_endpoint: str) -> List:
    return [handle_record_entry(record, blob_sas_url, florence_endpoint) for record in record_list]


def handle_record_entry(record: Dict, blob_sas_url: str, florence_endpoint: str) -> Dict:
    url = extract_url(record)
    record_id = record[RECORD_ID_TAG]
    filename = get_file_name_without_extension(url)
    entry_name = f"{filename}_{record_id}"

    document_info = inferencing(
        florence_endpoint, entry=entry_name, url=url, blob_sas_url=blob_sas_url)

    data_entry = format_data_entry(url, document_info)
    return format_entry(record_id=record_id, data_entry=data_entry)


def get_file_name_without_extension(url: str):
    filename, _ = os.path.splitext(os.path.basename(url))
    return filename


def format_entry(record_id: str, data_entry: Dict) -> Dict:
    return {RECORD_ID_TAG: record_id,
            DATA_TAG: data_entry,
            }


def extract_url(value: Dict) -> str:
    return value[DATA_TAG][URL_TAG]


def format_data_entry(url: str, document_info: List) -> Dict:
    return {URL_TAG: url,
            FRAMES_TAG: document_info,
            FRAME_COUNT_TAG: len(document_info)}


def format_http_response(response_body: Dict[str, Any]) -> func.HttpResponse:
    response = func.HttpResponse(json.dumps(
        response_body, default=lambda obj: obj.__dict__))
    response.headers[CONTENT_TYPE_TAG] = CONTENT_TYPE
    return response


def get_request_schema():
    return {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "type": "object",
        "properties": {
            "values": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        RECORD_ID_TAG: {"type": "string"},
                        DATA_TAG: {
                            URL_TAG: {"type": "string"},
                        },
                    },
                    "required": ["recordId", "data"],
                },
            }
        },
        "required": ["values"],
    }
