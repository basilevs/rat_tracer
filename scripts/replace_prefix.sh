#!/bin/bash

# 1. Capture arguments
OLD_PREFIX="$1"
NEW_PREFIX="$2"

# 2. Validation: Ensure both arguments are provided
if [ -z "$OLD_PREFIX" ] || [ -z "$NEW_PREFIX" ]; then
    echo "Error: Missing arguments."
    echo "Usage: ./replace_prefix.sh <old_prefix> <new_prefix>"
    exit 1
fi

# 3. Validation: Ensure Old Prefix isn't empty to prevent accidental mass renaming
if [ "$OLD_PREFIX" == "" ]; then
    echo "Error: Old prefix cannot be an empty string."
    exit 1
fi

SCRIPT_NAME=$(basename "$0")

# 4. Find and Loop
# -name "${OLD_PREFIX}*": Only finds files starting with the old prefix.
# This ensures we do not touch files that don't match.
find . -maxdepth 1 -type f -name "${OLD_PREFIX}*" -print0 | while IFS= read -r -d '' file; do

    # Strip the leading "./" given by find
    FILENAME="${file#./}"

    # Safety: Don't rename the script itself if it happens to match
    if [ "$FILENAME" == "$SCRIPT_NAME" ]; then
        continue
    fi

    # 5. String Manipulation
    # ${var#pattern} removes the shortest match of pattern from the front of string
    BASE_NAME="${FILENAME#$OLD_PREFIX}"
    
    # Construct the new name
    NEW_NAME="${NEW_PREFIX}${BASE_NAME}"

    # 6. Rename
    mv "$file" "$NEW_NAME"

done

echo "Operation complete."