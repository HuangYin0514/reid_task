####################################################################################
#
# Commond
#
####################################################################################

cp -rf ex_main1 ex_main1_cp 
rm -rf ex_main1/outputs/models/*
cp -rf ex_main1  version/tmp/ex16
cp ex_main1_cp/outputs/models/model_335.tar version/tmp/ex16/outputs/models/
rm -rf ex_main1_cp
