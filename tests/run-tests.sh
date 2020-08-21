#!/usr/bin/env bash

# set -x

RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RESET='\033[0m'

TMP_DIR=`mktemp -d`

TESTS_DIR=`dirname $0`
CASES_DIR="$TESTS_DIR/test_cases"
RESULTS_DIR="$TESTS_DIR/test_results"

for CASE_PATH in $(find "$CASES_DIR" -regex '.*\.txt$'); do
    BASE_NAME=`basename $CASE_PATH`
    CASE_NAME=${BASE_NAME%.txt}
    RESULTS_PATH="$RESULTS_DIR/$CASE_NAME.png"

    echo "checking case $CASE_NAME"

    if [[ ! -f $RESULTS_PATH ]]; then
        echo -e "  ${CYAN}creating $RESULTS_PATH ${RESET}"
        curl `cat $CASE_PATH` > $RESULTS_PATH
        if [[ "$?" != "0" ]]; then exit 1; fi
    else 
        TMP_PATH=`mktemp -t example.XXXXXXXXXX -p "$TMP_DIR"`
        curl `cat $CASE_PATH` > "$TMP_PATH"
        if [[ "$?" != "0" ]]; then exit 1; fi
        if cmp -s "$TMP_PATH" $RESULTS_PATH; then
            echo -e "  ${BLUE}cases matched: $CASE_NAME ${RESET}"
        else
            echo -e "  ${RED}cases mismatched: $CASE_NAME ${RESET}"
        fi
    fi
done

rm -r "$TMP_DIR"
