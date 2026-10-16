from fastapi import FastAPI
from faq_app.api.routes import router
from faq_app.config import Config

app = FastAPI(title=Config.APP_NAME, version=Config.VERSION)

app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)
