SHELL := /bin/bash

.PHONY: build gpu-check help shell tensorboard

build:
	docker compose build

gpu-check:
	docker compose run --rm sam-clip gpu-check

help:
	docker compose run --rm sam-clip help

shell:
	docker compose run --rm sam-clip shell

tensorboard:
	docker compose up tensorboard
