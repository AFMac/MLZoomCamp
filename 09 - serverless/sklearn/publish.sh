ECR_URL="343644158184.dkr.ecr.us-west-2.amazonaws.com"

aws ecr get-login-password \
  --region "us-west-2" \
| docker login \
  --username AWS \
  --password-stdin ${ECR_URL}

REMOTE_IMAGE_TAG="${ECR_URL}/churn-prediction-lambda:v1"

docker build -t churn-prediction-lambda .
docker tag churn-prediction-lambda ${REMOTE_IMAGE_TAG}
docker push ${REMOTE_IMAGE_TAG}

docker run -it --rm -v $(pwd)/models:/models agrigorev/tensorflow-onnx-runtime

docker run -it --rm -v $(pwd):/task agrigorev/model-2025-hairstyle:v1
docker run -it --rm \
  -v $(pwd)/models:/models \
  tensorflow-onnx-runtime:$TAG