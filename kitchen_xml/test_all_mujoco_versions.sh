#!/bin/bash

versions=(
	"2.1.3"
	"2.1.4"
	"2.1.5"
	"2.2.0"
	"2.2.1"
	"2.2.2"
	"2.3.0"
	"2.3.1"
	"2.3.2"
	"2.3.3"
	"2.3.5"
	"2.3.6"
	"2.3.7"
	"3.0.0"
	"3.0.1"
	"3.1.0"
	"3.1.1"
	"3.1.2"
	"3.1.3"
	"3.1.4"
	"3.1.5"
	"3.1.6"
)

for version in "${versions[@]}"; do
	echo "Testing mujoco==$version"
	pip install mujoco==$version
	python verify_no_changes.py
done
