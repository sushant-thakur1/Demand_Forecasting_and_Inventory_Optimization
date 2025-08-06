#!/bin/bash

bash retail_directory.sh
bash retail_put_data.sh

hive -f retail_ddl.hive
hive -f retail_load.hive
hive -f retail_query.hive
