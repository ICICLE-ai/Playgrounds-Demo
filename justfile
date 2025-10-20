# https://just.systems

default:
    echo 'Hello, world!'

run:
  nerdctl run -t -p 3000:5000 -v "./prompts:/app/prompts/" --name playgrounds-demo --env-file .env ghcr.io/icicle-ai/playgrounds-demo:latest

build:
  nerdctl build --platform linux/amd64 -t ghcr.io/icicle-ai/playgrounds-demo:latest .

stop:
  nerdctl stop playgrounds-demo

rm:
  nerdctl rm playgrounds-demo
