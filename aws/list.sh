export AWS_PROFILE=pfc
aws ec2 describe-instances |
	jq '.Reservations'|
	jq 'map(.Instances[0])'|
	jq 'map(.|select(.State.Name != "terminated"))'|
	jq 'map(.|select(.Tags|map(.|select(.Key == "Name").Value=="pfc-kamikaze-run")))'|
	jq 'map((.Tags[]|select(.Key == "Name")).Value+": "+.InstanceId)[]'|
	sed 's/"//g'


