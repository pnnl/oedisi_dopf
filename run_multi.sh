#!/bin/bash
scenario="$1"

if [ "$scenario" = "" ];
then
	echo You must enter a scenario name
else
	oedisi build -m --component-dict scenario/$scenario/components.json --system scenario/$scenario/system.json --target-directory build_$scenario
	docker compose up
fi
