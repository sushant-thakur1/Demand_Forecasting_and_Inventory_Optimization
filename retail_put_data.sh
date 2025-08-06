#!/bin/bash

hdfs dfs -put ~/shared/product_data.csv /user/talentum/Final_Project
hdfs dfs -put ~/shared/store_data.csv /user/talentum/Final_Project
hdfs dfs -put ~/shared/train.csv /user/talentum/Final_Project

hdfs dfs -ls /user/talentum/Final_Project

