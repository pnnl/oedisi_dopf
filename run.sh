#!/bin/bash
scenario="$1"

if [ "$scenario" = "" ];
then
	echo You must enter a scenario name
else
	if [ ! -d "outputs/$scenario" ]; then
   		echo "Creating $scenario directory"
   		mkdir -p "outputs/$scenario"
	fi
	pkill -9 helics_broker
	pkill -9 python
	oedisi build --component-dict scenarios/$scenario/components.json --system scenarios/$scenario/system.json --target-directory builds/$scenario
	oedisi run --runner builds/$scenario/system_runner.json
	#oedisi evaluate-estimate --path outputs/$scenario --metric MARE
fi
