import json
import logging
import os
import random
import string
import time
from typing import Dict, List

import requests

from dataclasses import dataclass

from azure.storage.blob import BlobServiceClient, ContainerClient

SUBSCRIPTION_ID_TAG = 'apim-subscription-id'
SUBSCRIPTION_ID = 'indexer'
CONTENT_TYPE_TAG = 'Content-Type'
CONTENT_TYPE = 'application/json'
CONTAINER_NAME = 'framesaux'  # avoid duplication with cognitive_indexer.py
# avoid duplication with cognitive_indexer.py
CONNECTION_STRING = 'BLOB_CONNECTION_STRING'


@dataclass
class Record:
    record_id: str
    url: str
    entry_name: str

    def __post_init__(self):
        self.entry_name = f"{self.entry_name}_{generate_random_string()}"


@dataclass
class Frame:
    presentation_timestamp: int
    presentation_timestamp_in_ms: int
    duration_in_ms: int
    embeddings: List[float]
    frame_id: int
    interval_id: int
    embeddings_metadata: Dict


def inferecing_over_several_entries(florence_endpoint: str, records: List[Record], blob_sas_url: str) -> List[List[Frame]]:
    (put_file_into_florence(florence_endpoint=florence_endpoint, entry=record.entry_name,
     subscription_id=SUBSCRIPTION_ID, blob_file_url=record.url, blob_sas_url=blob_sas_url) for record in records)
    [pull_result(florence_endpoint=florence_endpoint, entry=record.entry_name,
                 subscription_id=SUBSCRIPTION_ID) for record in records]
    downloaded_data = [download_result_from_container(
        record.entry_name) for record in records]
    return [handle_downloaded_frames_data(data) for data in downloaded_data]


def inferencing(florence_endpoint: str, entry: str, url: str, blob_sas_url: str) -> List[Frame]:
    entry_name = f"{entry}_{generate_random_string()}"
    run_inference(florence_endpoint, entry_name,
                  SUBSCRIPTION_ID, url, blob_sas_url)

    data = download_result_from_container(entry_name)
    return handle_downloaded_frames_data(data)


def call_florence_to_records(florence_endpoint: str, records: List[Record], blob_sas_url: str) -> List[Record]:
    (put_file_into_florence(florence_endpoint=florence_endpoint, entry=record.entry_name,
     subscription_id=SUBSCRIPTION_ID, blob_file_url=record.url, blob_sas_url=blob_sas_url) for record in records)
    [pull_result(florence_endpoint=florence_endpoint, entry=record.entry_name,
                 subscription_id=SUBSCRIPTION_ID) for record in records]


def download_data_from_records(records: List[Record]) -> List[List[Frame]]:
    downloaded_data = [download_result_from_container(
        record.entry_name) for record in records]
    return [handle_downloaded_frames_data(data) for data in downloaded_data]


def run_inference(florence_endpoint: str, entry: str, subscription_id: str, url: str, blob_sas_url: str):
    put_file_into_florence(
        florence_endpoint,
        entry,
        subscription_id,
        url,
        blob_sas_url
    )

    pull_result(florence_endpoint=florence_endpoint,
                entry=entry, subscription_id=subscription_id)


def connect_to_container(container_name: str, connection_string: str):
    blob_service_client = BlobServiceClient.from_connection_string(
        connection_string)
    container_client = blob_service_client.get_container_client(
        container_name)
    return container_client


def download_from_container_based_on_partial_path(container_client: ContainerClient, partial_path: str) -> List:
    blobs = container_client.list_blobs(name_starts_with=partial_path)
    files = []
    for blob in blobs:
        if not blob.name.endswith(".json"):
            continue
        blob_client = container_client.get_blob_client(blob)
        blob_data = blob_client.download_blob()
        content = json.loads(blob_data.content_as_text())
        files.append(content)
    return files


def pull_result(florence_endpoint: str, entry: str, subscription_id: str):
    get_endpoint = f"{florence_endpoint}/status/{entry}"
    headers = create_headers(subscription_id)
    response = ""
    for i in range(1, 10):
        response = requests.get(get_endpoint, headers=headers)
        if not response.ok:
            response.raise_for_status()

        response_body = response.json()
        state = response_body['state']
        logging.info(f"Response: {response_body}")
        if state in ['success', 'completed']:
            return response_body
        if state in ['running', 'queued']:
            time.sleep(15 * (2 ** i))
            continue
    raise RuntimeError(f"Not enought time. Last content: {response.content}")


def put_file_into_florence(florence_endpoint: str, entry: str, subscription_id: str, blob_file_url: str, blob_sas_url: str):
    endpoint = f"{florence_endpoint}/imageretrieval/{entry}"
    headers = create_headers(subscription_id)
    request_payload = create_florence_payload(blob_file_url, blob_sas_url)
    logging.info(f"Request: {request_payload}")
    logging.info(f"Endpoint: {endpoint}")
    logging.info(f"Headers: {headers}")
    response = requests.put(endpoint, headers=headers,
                            data=json.dumps(request_payload))
    if not response.ok:
        logging.info(f"Response: {response.content}")
        response.raise_for_status()

    response_payload = response.json()
    logging.info(f"Response: {response_payload}")


def create_florence_payload(blob_file_url: str, blob_sas_url: str) -> Dict:
    return {
        'operation': {
            'action': 'featurizeimage'
        },
        'input': {
            'kind': 'inline',
            'files': [
                    {
                        'uri': blob_file_url,
                        'mediaType': 'video'
                    }
            ]
        },
        'output': {
            'kind': 'blob',
            "sasUri": blob_sas_url,
            'authentication': {'kind': 'blobSas'}
        }
    }


def create_headers(subscription_id: str) -> Dict:
    return {
        SUBSCRIPTION_ID_TAG: subscription_id,
        CONTENT_TYPE_TAG: CONTENT_TYPE
    }


def generate_random_string(length=6) -> str:
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(length))


def download_result_from_container(entry: str) -> List:
    blob_connection_string = os.getenv(CONNECTION_STRING)
    container = connect_to_container(CONTAINER_NAME, blob_connection_string)
    return download_from_container_based_on_partial_path(container, f"{entry}")


def handle_downloaded_frames_data(results: List[Dict]) -> List[Frame]:
    try:
        info = []
        for result in results:
            content_list = result['result']
            for entry in content_list:
                insights = entry['insights']['results']
                for insight in insights:
                    interval_insights = insight['interval_insights']
                    for interval_insight in interval_insights.values():
                        interval_id = interval_insight['interval_id']
                        for interval_frame_insight in interval_insight['interval_frame_insights']:
                            frame_id = interval_frame_insight['frame_id']
                            presentation_timestamp = interval_frame_insight['presentation_timestamp']
                            presentation_timestamp_in_ms = interval_frame_insight[
                                'presentation_timestamp_in_ms']
                            duration_in_ms = interval_frame_insight['duration_in_ms']
                            embeddings_metadata = interval_frame_insight['frame_insights'][0]['metadata']
                            embeddings = embeddings_metadata['Vector']
                            del embeddings_metadata['Vector']
                            frame = Frame(
                                presentation_timestamp=presentation_timestamp,
                                presentation_timestamp_in_ms=presentation_timestamp_in_ms,
                                duration_in_ms=duration_in_ms,
                                embeddings=embeddings,
                                frame_id=frame_id,
                                interval_id=interval_id,
                                embeddings_metadata=embeddings_metadata
                            )
                            info.append(frame)
        return info
    except Exception as e:
        logging.error(f"Error: {e} -> Content: {results}")
        raise ValueError(f"Error: {e} -> Content: {results}")
