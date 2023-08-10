#!/bin/bash
scenario="$1"

if [ "$scenario" = "" ];
then
	echo You must enter a scenario name
else
	pkill -9 helics_broker
	pkill -9 python
	oedisi build --component-dict scenario/$scenario/components.json --system scenario/$scenario/system.json --target-directory build_$scenario
	oedisi run --runner build_$scenario/system_runner.json
fi
