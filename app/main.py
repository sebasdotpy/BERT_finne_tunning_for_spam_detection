import os
import sys
from transformers import BertTokenizer

import uvicorn
from fastapi import FastAPI, Request, status
from fastapi.logger import logger
from fastapi.encoders import jsonable_encoder
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

import torch

from model import Model
from predict import predict
from config import CONFIG
from exception_handler import validation_exception_handler, python_exception_handler
from schema import *

# Initialize API Server
app = FastAPI(
    title="ML Model",
    description="Description of the ML Model",
    version="0.0.1",
    terms_of_service=None,
    contact=None,
    license_info=None
)

# Allow CORS for local debugging
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Mount static folder, like demo pages, if any
app.mount("/static", StaticFiles(directory="static/"), name="static")

# Load custom exception handlers
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, python_exception_handler)


@app.on_event("startup")
async def startup_event():
    """
    Initialize FastAPI and add variables
    """

    logger.info('Running envirnoment: {}'.format(CONFIG['ENV']))
    logger.info('PyTorch using device: {}'.format(CONFIG['DEVICE']))

    # Initialize the pytorch model
    model = Model
    # model.load_state_dict(
    #     torch.load(CONFIG['MODEL_PATH'], map_location=torch.device(CONFIG['DEVICE']))
    # )
    try:
        model.cuda()
    except:
        pass
    model.eval()

    # add model and other preprocess tools too app state
    app.package = {
        "tokenizer": BertTokenizer.from_pretrained(
                CONFIG["TOKENIZER_PATH"],
                do_lower_case=True
            ),
        "model": model
    }


@app.post('/api/v1/predict',
          response_model=InferenceResponse,
          responses={422: {"model": ErrorResponse},
                     500: {"model": ErrorResponse}}
          )
def do_predict(request: Request, body: InferenceInput):
    """
    Perform prediction on input data
    """

    logger.info('API predict called')
    logger.info(f'input: {body}')

    # prepare input data
    X = body.input_msg

    # run model inference
    y = predict(app.package, X)

    # prepare json for returning
    results = {
        "result": y
    }

    logger.info(f'results: {results}')

    return {
        "error": False,
        "results": results
    }


@app.get('/about')
def show_about():
    """
    Get deployment information, for debugging
    """

    def bash(command):
        output = os.popen(command).read()
        return output

    return {
        "sys.version": sys.version,
        "torch.__version__": torch.__version__,
        "torch.cuda.is_available()": torch.cuda.is_available(),
        "torch.version.cuda": torch.version.cuda,
        "torch.backends.cudnn.version()": torch.backends.cudnn.version(),
        "torch.backends.cudnn.enabled": torch.backends.cudnn.enabled,
        "nvidia-smi": bash('nvidia-smi')
    }


if __name__ == '__main__':
    # server api
    uvicorn.run("main:app", host="127.0.0.1", port=8080,
                reload=True, debug=True, log_config="log.ini"
                )