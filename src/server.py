import os
from fastapi import FastAPI, UploadFile, File, Response
import uvicorn
import logging
from src.ModelSingleton import SAMModel, init_model
from src.ApiDataStructs import ModelResponse


app = FastAPI()
logger = logging.getLogger('gunicorn.error')


class OctetStreamResponse(Response):
    media_type = "application/octet-stream"


@app.on_event("startup")
def load_info():
    config_file_path = os.environ.get('CONFIG_PATH',
                                      os.path.join(
                                          os.path.dirname(os.path.realpath(__file__)),
                                          "..",
                                          'config.yaml'
                                      )
                                      )
    init_model(config_file_path)


@app.get("/models/")
async def get_model_names() -> ModelResponse:
    return ModelResponse(name=SAMModel.model_name)


@app.post("/image_slice_embedding", response_class=OctetStreamResponse)
async def get_image_slice_embedding(image: UploadFile = File(...)):
    SAMModel.set_uploaded_image(image.file)
    image_embedding = SAMModel.model_predictor.get_image_embedding().cpu().numpy()
    embedding_bytes = image_embedding.tobytes()
    return Response(content=embedding_bytes,
                    headers={
                        'Content-Length': len(embedding_bytes),
                        'X-Numpy-Dtype': str(image_embedding.dtype),
                        'X-Numpy-Shape': str(image_embedding.shape)
                    },
                    media_type="application/octet-stream")


if __name__ == '__main__':
    uvicorn.run(app, port=8080)
