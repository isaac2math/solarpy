#!/bin/bash

TeX_version_control () {

	mkdir raw_results

	for folder_name in application example_holdout simul_lasso_solar demo raw_results example_IRC simul_bolasso_bsolar
	do
		html_files=$(find ./$folder_name/ -iname "*.html")	
		cp -rf $html_files ./raw_results/
	done

	find ./raw_results/ -type f -name '*checkpoint*' -delete

	mkdir ../version_control
	current_time=$(date "+%m%d_%H%M")
	new_zipName="v"$current_time"_Ning.zip"

	echo
	echo 'after press Enter, input the revision summary of this version, press enter and press CTRL + D.' 
	read -p 'Plz start a new line for a different entry and conclude the line by ;'
	echo
	echo "" >> revision_summary.txt
	echo "version : $(date "+%Y_%m%d_%H:%M")" >> revision_summary.txt
	echo "" >> revision_summary.txt
	echo "" >> revision_summary.txt
	cat >> revision_summary.txt
	echo "" >> revision_summary.txt
	zip -r $new_zipName ./*
	
	mv ./*.zip ../version_control/

	#make a backup directory
	#new_folderName="v"$current_time"_Ning"
	#mkdir ../$new_folderName
	#cp -r ./* ../$new_folderName

	#update the display version of the revision summary 
	cp ./revision_summary.txt ../revision_summary.txt

}

TeX_version_control
