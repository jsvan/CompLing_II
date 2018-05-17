#!/bin/bash
git rm -r $1
git commit -m "renoved $1"
git push
