#!/bin/bash

bug_check_py () {

	local file_name=$(ls -d */*.py)

	for name in $file_name
	do
		echo
		echo "checking the potential bugs of the py file : $name" 
		echo

		python $name
		
		echo
		read -p "Press enter to continue"
		echo
	done

}

bug_check_py
