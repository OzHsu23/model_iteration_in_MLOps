import argparse
import os
import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument("--setting_path", type=str, default="setting.json")
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--port", type=int, default=8020)
args = parser.parse_args()

os.environ["SETTING_PATH"] = args.setting_path

uvicorn.run("app.app:app", host=args.host, port=args.port)
