import os

from fastapi import FastAPI
import uvicorn
import logging
import yaml

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
    with open(config_file_path) as config_stream:
        config = yaml.load(config_stream, Loader=yaml.SafeLoader)
        logger.info(config)


@app.get("/models/")
async def get_model_names():
    """
    """
    return list(['ViT_H SAM model'])


if __name__ == '__main__':
    uvicorn.run(app, port=8080)
