import azure.functions as func

from indexer.cognitive_indexer import extract_video_embeddings

app = func.FunctionApp()


@app.function_name(name="CognitiveIndexer")
@app.route(route="blob-indexer")
def blob_indexing(req: func.HttpRequest) -> func.HttpResponse:
    return extract_video_embeddings(req)
