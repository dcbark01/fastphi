INFERENCE_HOST=0.0.0.0
INFERENCE_PORT=8000

start:  ## Start the inference server
	@echo "Starting the inference server"
	uvicorn app:app --host "$(INFERENCE_HOST)" --port "$(INFERENCE_PORT)"

clean:  ## Remove extraneous files
	@echo "Removing logs"
	@find ./ -type f \( -name "assertion.log" -o -name "azure_openai_usage.log" -o -name "openai_usage.log" \) -exec rm {} +
	@echo "Removing pycache files"
	@find ./ \( -type f -name '*.pyc' -o -type d -name '__pycache__' \) -exec rm -rf {} +

.PHONY: help
.DEFAULT_GOAL := help

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'