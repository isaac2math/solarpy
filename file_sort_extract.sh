#!/bin/bash

file_sort_extract () {
	for folder_name in application example_test simul_lasso_solar demo example_IRC simul_bolasso_bsolar
	do
		#delete checkpoints
		delete_file=$(find ./$folder_name/ -iname "*-checkpoint.*")	
		rm -rf $delete_file
		
		#convert to HTML
		#ipynb_file=$(find ./$folder_name/ -iname "*.ipynb")	
		#jupyter nbconvert --to html $ipynb_file

		#extract all raw results
		#extract_file=$(find ./$folder_name/ -iname "*.html")	
		#mv $extract_file ./raw_results/
	done
}

file_sort_extract