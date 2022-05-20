#!/bin/bash

bug_check_py () {

	local file_name=$(ls *.py)

	for name in $file_name
	do
		echo
		echo "checking the potential bugs of the py file : $name" 
		echo

		python $name
	done

}

bug_check_py

