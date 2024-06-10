# Copyright 2021 BBC
# Authors: Chris Newell <chris.newell@bbc.co.uk>
#
# License: Apache-2.0

import json

from citron.utils import get_parser
from citron.citron import Citron
from citron.logger import logger

from typing_extensions import Annotated
from fastapi import FastAPI, Form, Response, status

nlp = get_parser(use_gpu = True, use_small = False)
citron = Citron("./models/en_2021-11-15", nlp=nlp)

app = FastAPI()

@app.post("/quotes")
async def entities(text: Annotated[str, Form()], response: Response):
    # raw_data = await request.body()
    # data = raw_data.decode('utf-8')
    # string_entities = extract_entities(data)
    # return Response(content=string_entities, media_type="application/json")
    if text is None or text.strip() == "":
        response.status_code = status.HTTP_400_BAD_REQUEST
        response.headers["Content-Type"] = "application/json"
        return {"error": "A text parameter must be provided."}
    
    try:
        results = citron.extract(text)
    except ValueError as err:
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        response.headers["Content-Type"] = "application/json"
        return { "error": err.message }
    
    response.headers["Content-Type"] = "application/json; charset=utf-8"
    return results
