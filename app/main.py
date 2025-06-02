import nest_asyncio
nest_asyncio.apply()

from fastapi import FastAPI
import uvicorn
from route.image import router as image_router
from route.video import router as video_router

app = FastAPI()

# Include routers
app.include_router(image_router, prefix="/image", tags=["image"])
app.include_router(video_router, prefix="/video", tags=["video"])

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8185)