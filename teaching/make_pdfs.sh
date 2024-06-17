#!/bin/bash

# jupyter nbconvert --execute --to html --output temp.html predictive_modeling_pipeline.ipynb 
# wkhtmltopdf temp.html predictive_modeling_pipeline.pdf 

# jupyter nbconvert --execute --to html --output temp.html selecting_the_best_model.ipynb 
# wkhtmltopdf temp.html selecting_the_best_model.pdf 

# jupyter nbconvert --execute --to html --output temp.html linear_models.ipynb 
# wkhtmltopdf temp.html linear_models.pdf 

cd 2024-sicss
jupyter nbconvert --execute --to html --output temp.html data-exploration.ipynb 
wkhtmltopdf temp.html data-exploration.pdf 

jupyter nbconvert --execute --to html --output temp.html categorical-encoding.ipynb 
wkhtmltopdf temp.html categorical-encoding.pdf 

rm -rf temp.html
