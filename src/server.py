import os
import json
import numpy as np
from fastapi import FastAPI
from fastapi import HTTPException
import uvicorn
import logging
from src.ConfigSingleton import init_configuration, SAMConfig
from src.ApiDataStructs import ModelResponse


app = FastAPI()
logger = logging.getLogger('gunicorn.error')


@app.on_event("startup")
def load_info():
    config_file_path = os.environ.get('CONFIG_PATH',
                                      os.path.join(
                                          os.path.dirname(os.path.realpath(__file__)),
                                          "..",
                                          'config.yaml'
                                      )
                                      )
    init_configuration(config_file_path)


@app.get("/models/")
async def get_model_names() -> ModelResponse:
    return {'name': SAMConfig.models['name']}


@app.get("/data_info/")
async def get_data_info():
    ret = []
    for key, val in SAMConfig.data.items():
        ret.append({
            'name': key,
            'total_slices': val['total_slices']
        })
    return ret


@app.get("/data_slice_embedding/{data_name}/{slice_no}")
async def get_data_slice_embedding(data_name: str, slice_no: int):
    if data_name not in SAMConfig.data:
        raise HTTPException(status_code=404, detail="input data name not found")
    embed_file = SAMConfig.data[data_name]['embedding'].format(slice_no)
    if not os.path.isfile(embed_file):
        raise HTTPException(status_code=404, detail="input slice_no is invalid")
    embed_data = np.load(embed_file)
    return json.dumps(embed_data.tolist())


if __name__ == '__main__':
    uvicorn.run(app, port=8080)
