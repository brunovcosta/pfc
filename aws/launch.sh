#export AWS_PROFILE=pfc
aws ec2 run-instances --cli-input-json file://instance.json --user-data file://setup_50_sepcnn.sh
