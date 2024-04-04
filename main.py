import uvicorn
from dotenv import load_dotenv

load_dotenv()

HOST = '0.0.0.0'

if __name__ == "__main__":
    uvicorn.run("server.api:app", host=HOST, port=8000, reload=True)