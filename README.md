# README.md

1. Prepare your code and prediction models
2. Fill your team name and email into `submitter.json`
3. Create your Dockerfile(Please refer to Dockerfile.example)
4. Build a docker images from the Dockerfile by run `docker build -t <Your-image-name> .`
5. Execute the test command `docker run -it --rm --gpus device=0 -v "/your/input/folder/":"/input" -v "/your/output/folder/":"/output" <Your-image-name>`
