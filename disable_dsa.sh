#!/bin/bash

if [ -z "$1" ]; then
   echo "USAGE: $0 all|dsa[0-9]*"
   exit
fi

set -x

if [ "$1" == "all" ]; then
   for id in {0..14..2}
   do
      accel-config disable-device dsa${id}
   done;
else
   accel-config disable-device $1
fi

