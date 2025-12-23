#!/bin/bash
scenario="$1"

if [ "$scenario" = "" ];
then
	echo You must enter a scenario name
else
	#oedisi build -m --component-dict scenarios/$scenario/components.json --system scenarios/$scenario/system.json --target-directory builds/multi/$scenario
	oedisi build -m --component-dict scenarios/$scenario/components.json --system scenarios/$scenario/system.json --target-directory build
	#docker compose up
fi
