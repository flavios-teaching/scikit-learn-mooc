#!/bin/bash

jupyter nbconvert --to html --output temp.html predictive_modeling_pipeline.ipynb 
wkhtmltopdf temp.html predictive_modeling_pipeline.pdf 

jupyter nbconvert --to html --output temp.html selecting_the_best_model.ipynb 
wkhtmltopdf temp.html selecting_the_best_model.pdf 

rm -rf temp.html
