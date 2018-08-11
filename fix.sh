#!/bin/sh

git filter-branch --env-filter '
OLD_EMAIL="ubuntu@ip-172-31-2-124.sa-east-1.compute.internal"
CORRECT_NAME="Bruno Vieira Costa"
CORRECT_EMAIL="brunovcosta@outlook.com"
if [ "$GIT_COMMITTER_EMAIL" = "$OLD_EMAIL" ]
then
    export GIT_COMMITTER_NAME="$CORRECT_NAME"
    export GIT_COMMITTER_EMAIL="$CORRECT_EMAIL"
fi
if [ "$GIT_AUTHOR_EMAIL" = "$OLD_EMAIL" ]
then
    export GIT_AUTHOR_NAME="$CORRECT_NAME"
    export GIT_AUTHOR_EMAIL="$CORRECT_EMAIL"
fi
' --tag-name-filter cat -- --branches --tags
