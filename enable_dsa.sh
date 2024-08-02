#!/bin/bash
set -v

accel-config config-wq --group-id=0 --mode=shared --wq-size=128 --threshold=127 --type=user --priority=10 --name="app-0" --driver-name="user" --max-transfer-size $((2*1024*1024*1024)) --block-on-fault=0 dsa0/wq0.0
accel-config config-engine dsa0/engine0.0 --group-id=0
accel-config enable-device dsa0
accel-config enable-wq dsa0/wq0.0

accel-config config-wq --group-id=0 --mode=shared --wq-size=128 --threshold=127 --type=user --priority=10 --name="app-1" --driver-name="user" --max-transfer-size $((2*1024*1024*1024)) --block-on-fault=0 dsa2/wq2.0
accel-config config-engine dsa2/engine2.0 --group-id=0
accel-config enable-device dsa2
accel-config enable-wq dsa2/wq2.0

accel-config config-wq --group-id=0 --mode=shared --wq-size=128 --threshold=127 --type=user --priority=10 --name="app-2" --driver-name="user" --max-transfer-size $((2*1024*1024*1024)) --block-on-fault=0 dsa4/wq4.0
accel-config config-engine dsa4/engine4.0 --group-id=0
accel-config enable-device dsa4
accel-config enable-wq dsa4/wq4.0

accel-config config-wq --group-id=0 --mode=shared --wq-size=128 --threshold=127 --type=user --priority=10 --name="app-3" --driver-name="user" --max-transfer-size $((2*1024*1024*1024)) --block-on-fault=0 dsa6/wq6.0
accel-config config-engine dsa6/engine6.0 --group-id=0
accel-config enable-device dsa6
accel-config enable-wq dsa6/wq6.0

accel-config config-wq --group-id=0 --mode=shared --wq-size=128 --threshold=127 --type=user --priority=10 --name="app-4" --driver-name="user" --max-transfer-size $((2*1024*1024*1024)) --block-on-fault=0 dsa8/wq8.0
accel-config config-engine dsa8/engine8.0 --group-id=0
accel-config enable-device dsa8
accel-config enable-wq dsa8/wq8.0

accel-config config-wq --group-id=0 --mode=shared --wq-size=128 --threshold=127 --type=user --priority=10 --name="app-5" --driver-name="user" --max-transfer-size $((2*1024*1024*1024)) --block-on-fault=0 dsa10/wq10.0
accel-config config-engine dsa10/engine10.0 --group-id=0
accel-config enable-device dsa10
accel-config enable-wq dsa10/wq10.0

accel-config config-wq --group-id=0 --mode=shared --wq-size=128 --threshold=127 --type=user --priority=10 --name="app-6" --driver-name="user" --max-transfer-size $((2*1024*1024*1024)) --block-on-fault=0 dsa12/wq12.0
accel-config config-engine dsa12/engine12.0 --group-id=0
accel-config enable-device dsa12
accel-config enable-wq dsa12/wq12.0

accel-config config-wq --group-id=0 --mode=shared --wq-size=128 --threshold=127 --type=user --priority=10 --name="app-7" --driver-name="user" --max-transfer-size $((2*1024*1024*1024)) --block-on-fault=0 dsa14/wq14.0
accel-config config-engine dsa14/engine14.0 --group-id=0
accel-config enable-device dsa14
accel-config enable-wq dsa14/wq14.0

chmod -R a+r+w+x /dev/dsa/wq0.0
chmod -R a+r+w+x /dev/dsa/wq2.0
chmod -R a+r+w+x /dev/dsa/wq4.0
chmod -R a+r+w+x /dev/dsa/wq6.0

chmod -R a+r+w+x /dev/dsa/wq8.0
chmod -R a+r+w+x /dev/dsa/wq10.0
chmod -R a+r+w+x /dev/dsa/wq12.0
chmod -R a+r+w+x /dev/dsa/wq14.0

rdmsr 0xc8b
